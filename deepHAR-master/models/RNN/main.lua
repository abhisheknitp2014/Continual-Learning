require 'nn'
require 'rnn'
require 'optim'
require 'meanF1score'
require 'explib'
require 'hdf5'
json = require 'json'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('LSTM network for HAR')
cmd:text()
cmd:text('Options')
cmd:option('-seed',                   123,    'Initial random seed')
cmd:option('-logdir',                 'exp',  'Path to store model progress, results, and log file')
cmd:option('-data',                   '',     'Data-set to run on (DP datasource)')
cmd:option('-gpu',                    0,      'GPU to run on (default: 0)')
cmd:option('-cpu',                    false,  'Run on CPU')
cmd:option('-numLayers',              1,      'Number of LSTM layers')
cmd:option('-layerSize',              64,     'Number of cells in LSTM')
cmd:option('-learningRate',           0.1,    'Learning rate')
--cmd:option('-dropout',                0.5,    'Dropout (dropout == 0 -> no dropout)')
cmd:option('-momentum',               0.9,    'Momentum')
cmd:option('-learningRateDecay',      5e-5,   'Learning rate decay')
cmd:option('-maxInNorm',              2,      'Max-in-norm for regularisation')
cmd:option('-maxOutNorm',             0,      'Max-out-norm for regularisation')
cmd:option('-patience',               10,     'Patience in early stopping')
cmd:option('-minEpoch',               20,     'Minimum number of epochs before check for convergence')
cmd:option('-maxEpoch',               150,     'Stop after this number of epochs even if not converged')
cmd:option('-batchSize',              64,     'Batch-size (number of sequences in each batch)')
cmd:option('-stepSize',               16,      'Step-size when iterating through sequence')
cmd:option('-sequenceLength',         24,     'Sequence-length that is looked at in each batch')
cmd:option('-carryOverProb',          0.5,    'Probability to carry over hidden states between batches')
cmd:option('-ignore',                 false,  'Is there a class we should ignore?')
cmd:option('-ignoreClass',            0,      'Class to ignore for analysis')
cmd:option('-classWeights',           '',     'Weightings for classes. Must be string of weights separated with ","')
cmd:text()

-- parse input params
params = cmd:parse(arg)

if not params.cpu then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(params.gpu)
    cutorch.manualSeed(params.seed, params.gpu)
end

paths.mkdir(params.logdir)

-- create log file
cmd:log(params.logdir .. '/log', params)

-- preliminaries
torch.manualSeed(params.seed)
torch.setnumthreads(16)
epochPerformance = {} -- table to store progress
testPerformance = {}

-- Read in data-set and (maybe) store on GPU
f = hdf5.open(params.data, 'r')
data = f:read('/'):all()
f:close()
data.classes = json.load(params.data .. '.classes.json')

-- transfer data to gpu
if not params.cpu then
  data.training.inputs = data.training.inputs:cuda()
  data.test.inputs = data.test.inputs:cuda()
  data.validation.inputs = data.validation.inputs:cuda()
  data.training.targets = data.training.targets:cuda()
  data.test.targets = data.test.targets:cuda()
  data.validation.targets = data.validation.targets:cuda()
end

-- START model construction
model = nn.Sequential()

-- encoder
model:add(nn.Sequencer(nn.Linear(data.training.inputs:size(2), params.layerSize)))
-- recurrent layer
for i=1,params.numLayers do
  model:add(nn.Sequencer(nn.FastLSTM(params.layerSize, params.layerSize)))
  model:get(1+i):remember("both")
end
--model:add(nn.Sequencer(nn.FastLSTM(params.layerSize, params.layerSize)))
--model:get(2):remember("both") -- remember state in both training and evaluation
--model:get(3):remember("both") -- remember state in both training and evaluation
-- decoder
model:add(nn.Sequencer(nn.Linear(params.layerSize, #data.classes)))
-- classification
model:add(nn.Sequencer(nn.LogSoftMax()))
-- loss function
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- transfer to gpu
if not params.cpu then
  model:cuda()
  criterion:cuda()
end

-- get pointers for parameters
parameters, gradParameters = model:getParameters()

-- init weights
parameters:uniform(-0.08,0.08)

-- training function
function train(data, labels)
    epoch = epoch or 1
    print('Training epoch: ' .. epoch)
    model:training()

    -- make iterator for sampling sequences
    local batchIter = dataIter(data, labels)

    -- how many iterations in one epoch?
    local maxIter = math.floor((data:size(1)-params.sequenceLength)/(params.sequenceLength*params.batchSize))

    -- one pass through the data-set
    local cnt = 1
    for iter=1, maxIter do
        -- get x,y as tables with one entry for each timestep
        local x,y = batchIter()

        -- closure for gradient
        function feval(p)
            -- just in case
            collectgarbage()

            if p ~= parameters then
                parameters:copy(p)
            end
            gradParameters:zero()

            -- forward pass
            local outputs = model:forward(x)
            -- error
            local err = criterion:forward(outputs, y)
            -- backward pass
            local grad = criterion:backward(outputs, y)

            -- backward pass
            model:backward(x, grad)

            return loss, gradParameters
        end

        sgdState = sgdState or {
            learningRate = params.learningRate
        }

        optim.adagrad(feval, parameters, sgdState)

        -- renormalise weights
        for i,mod in ipairs(model.modules) do
            renormWeights(mod)
        end

        if math.random() > params.carryOverProb then
          model:get(2):forget()
        end

        xlua.progress(cnt, maxIter)
        cnt = cnt + 1
    end
    epoch = epoch + 1
end

-- test function
function test(data, labels, classes, isValidationSet)
    -- local vars
    local time = sys.clock()
    local confusion = optim.ConfusionMatrix(classes)
    confusion:zero()

    -- set to test mode
    model:evaluate()

    local perRun = 256

    -- test over given data
    print('<trainer> on testing Set:')
    local nbatch = torch.floor(labels:size(1) / perRun)
    local cnt = 1


    for x,y in linearDataIter(data, labels, perRun) do
        -- x,y are tables with an entry per input

        -- disp progress
        xlua.progress(cnt, nbatch)

        -- test samples
        local preds = model:forward(x)

        -- confusion:
        for i = 1, #x do
            confusion:add(preds[i][1], y[i][1])
        end

        cnt = cnt + 1
   end


    -- timing
    time = sys.clock() - time
    time = time / data:size(1)
    print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix
    print(confusion)

    local perf = {}
    perf.confusion = confusion
    perf.meanF1score = meanF1score(confusion)
    perf.TN = confusion.mat[1][1] / confusion.mat[1]:sum()
    perf.TP = confusion.mat[2][2] / confusion.mat[2]:sum()

    if isValidationSet then
        table.insert(epochPerformance, perf)
    else
        table.insert(testPerformance, perf)
    end

    print('meanF1score: ' .. meanF1score(confusion))
    return meanF1score(confusion)
end


-- main training loops
local best = 0
local progress = {}

progress.epochPerformance = epochPerformance
progress.testPerformance = testPerformance
for e=1,params.maxEpoch do
    train(data.training.inputs, data.training.targets)
    local score = test(data.validation.inputs, data.validation.targets, data.classes, true)
    local scoreT = test(data.test.inputs, data.test.targets, data.classes, false)

    torch.save('exp/test.dat', model)

    model:get(2):forget()

    if score > best then
        best = score
        epochPerformance.best = e
        torch.save(params.logdir .. '/model.dat', model)
    end

    -- save progress
    torch.save(params.logdir .. '/progress.dat', progress)

    if e > params.minEpoch then
        -- check for convergence
        if checkConvergence(e,params.patience) == true then
            break
        end
    end
end
