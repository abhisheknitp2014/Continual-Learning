import ray
from data import SubDataset

@ray.remote
def class_parallel(model, proc_id, new_classes, train_dataset, exemplars_per_class, args):
	print('class_parallel: proc_id: ', proc_id)
	chunk_size = int(len(new_classes) / args.parallel)

	start_from = proc_id * chunk_size
	if proc_id < (args.parallel) - 1:
		end_to = (proc_id+1) * chunk_size
	else:
		end_to = len(new_classes)
	print('class_parallel: proc_id: ', proc_id, ' start and End: ', start_from, ' ', end_to)
	for class_id in new_classes:
		if class_id >= start_from and class_id < end_to:
			# create new dataset containing only all examples of this class
			class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
			# based on this dataset, construct new exemplar-set for this class
			model.construct_exemplar_set(dataset=class_dataset, n=exemplars_per_class, class_id=class_id, args=args)

	# make exemplar set dict
	# self.exemplar_sets[class_id] = np.array(exemplar_set)
	print('class_parallel: proc_id: ', proc_id)
	print('rev_train: model.exemplar_dict: ', len(model.exemplar_dict))
	return model.exemplar_dict

@ray.remote
def constrcut_parallel(model, class_id, new_classes, train_dataset, exemplars_per_class, args):
	print('constrcut_parallel: class_id: ', class_id)
	chunk_size = int(args.parallel / len(new_classes))

	start_from = class_id * chunk_size
	if class_id < len(new_classes) - 1:
		end_to = (class_id+1) * chunk_size
	else:
		end_to = args.parallel

	print('constrcut_parallel: class_id: ', class_id, ' start and End: ', start_from, ' ', end_to)
	for class_id in new_classes:
		# create new dataset containing only all examples of this class
		class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
		# based on this dataset, construct new exemplar-set for this class
		model.construct_exemplar_set_parallel(dataset=class_dataset, n=exemplars_per_class, class_id=class_id, args=args, proc_id_l = list(range(start_from, end_to)))
