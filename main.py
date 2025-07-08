import argparse, glob, warnings, sys, random
from tools import *
from dataLoader import train_loader
from intmatch import ECAPAModel
from data_utils import *

def main_worker(args):

	if args.seed != None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		np.random.seed(args.seed)
	save_path = os.path.join(args.save_path, 'model')
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	else:
		print('already existing model: {}'.format(save_path))
	print("USE GPU: %s for training" % 0)
	## Define the data loader
	data_list, data_label = get_data(args.train_list, args.train_path)
	lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(data_list, data_label,
																lb_samples_per_class=args.lb_samples_per_class,
																num_classes=args.n_class)
	lb_trainloader = train_loader(args, lb_data, lb_targets, args.musan_path, args.rir_path, args.num_frames,
								  labelled=True)
	ulb_trainloader = train_loader(args, ulb_data, ulb_targets, args.musan_path, args.rir_path, args.num_frames,
								   labelled=False)
	lb_trainloader_sampler = torch.utils.data.BatchSampler(
		torch.utils.data.RandomSampler(lb_trainloader, replacement=True, num_samples=args.batch_size * args.max_it),
		batch_size=args.batch_size,
		drop_last=True)
	ulb_trainloader_sampler = torch.utils.data.BatchSampler(
		torch.utils.data.RandomSampler(ulb_trainloader, replacement=True, num_samples=args.batch_size * args.max_it * args.ratio),
		batch_size=args.batch_size * args.ratio,
		drop_last=True)

	Lb_trainloader = torch.utils.data.DataLoader(lb_trainloader, batch_sampler=lb_trainloader_sampler,
												 num_workers=args.n_cpu)
	Ulb_trainloader = torch.utils.data.DataLoader(ulb_trainloader, batch_sampler=ulb_trainloader_sampler,
												  num_workers=args.n_cpu)
	## Search for the exist models
	modelfiles = glob.glob('%s/model_0*.model' % args.model_save_path)
	modelfiles.sort()

	## Only do evaluation, the initial_model is necessary
	if args.eval == True:
		s = ECAPAModel(**vars(args))
		print("Model %s loaded from previous state!" % args.initial_model)
		s.load_parameters(args.initial_model)
		EER, minDCF = s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path)
		print("EER %2.2f%%, minDCF %.4f%%" % (EER, minDCF))
		quit()

	# Load pre-train model
	if args.initial_model != "":
		print("Model %s loaded from previous state!" % args.initial_model)
		s = ECAPAModel(**vars(args))
		s.load_parameters(args.initial_model)
	else:
		s = ECAPAModel(**vars(args))

	s.cuda()
	s.train_network(args, loader1=Lb_trainloader, loader2=Ulb_trainloader)

	quit()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="ECAPA_TDNN_Trainer")
	## Training Settings
	parser.add_argument('--num_frames', type=int, default=200,
						help='Duration of the input segments, eg: 200 for 2 second')
	parser.add_argument('--batch_size', type=int, default=150, help='Batch size')
	parser.add_argument('--ratio', type=int, default=1, help='Ratio for labeled and unlabeled data')
	parser.add_argument('--max_it', type=int, default=700000, help='Max iterations')
	parser.add_argument('--n_cpu', type=int, default=4, help='Number of loader threads')
	parser.add_argument('--test_step', type=int, default=7000, help='Test and save every [test_step] iterations')
	parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
	parser.add_argument("--lr_decay", type=float, default=0.97, help='Learning rate decay every [test_step] iterations')
	parser.add_argument("--tau", type=float, default=0.65, help='Initial intra-class threshold')
	parser.add_argument("--ema", type=float, default=0.999, help='Parameters of ema')

	parser.add_argument('--train_list', type=str, default="",
						help='The path of the training list, eg:"/speaker/voxceleb2/train_list.txt", which contains 1092009 lins')
	parser.add_argument('--train_path', type=str, default="",
						help='The path of the training data, eg:"speaker/voxceleb2/train/wav"')
	parser.add_argument('--eval_list', type=str, default="",
						help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
	parser.add_argument('--eval_path', type=str, default="",
						help='The path of the evaluation data, eg:"/speaker/voxceleb1/test/wav"')
	parser.add_argument('--musan_path', type=str, default="",
						help='The path to the MUSAN set, eg:"/speaker/Others/musan_split"')
	parser.add_argument('--rir_path', type=str, default="",
						help='The path to the RIR set, eg:"/speaker/Others/RIRS_NOISES/simulated_rirs"')
	parser.add_argument('--save_path', type=str, default="",
						help='Path to save the score.txt and models')
	parser.add_argument('--seed', default=None, type=int,
						help='random seed for initializing training.')
	parser.add_argument('--initial_model', type=str, default="",
						help='Path of the initial_model')

	## Model and Loss settings
	parser.add_argument('--C', type=int, default=1024, help='Channel size for the speaker encoder')
	parser.add_argument('--m', type=float, default=0.2, help='Loss margin in AAM softmax')
	parser.add_argument('--s', type=float, default=30, help='Loss scale in AAM softmax')
	parser.add_argument('--n_class', type=int, default=5994, help='Number of speakers')
	parser.add_argument('--lb_samples_per_class', type=float, default=0.4, help='lable samples per class')
	## Command
	parser.add_argument('--eval', dest='eval', action='store_true', help='Only do evaluation')
	parser.add_argument('--environment', type=bool, default=False, help='The environment of training is Windows')

	## Initialization
	warnings.simplefilter("ignore")
	torch.multiprocessing.set_sharing_strategy('file_system')
	args = parser.parse_args()
	args = init_args(args)
	n_gpus = torch.cuda.device_count()

	print('Python Version:', sys.version)
	print('PyTorch Version:', torch.__version__)
	if not torch.cuda.is_available():
		raise Exception('ONLY CPU TRAINING IS SUPPORTED')
	else:
		print('Number of GPUs:', torch.cuda.device_count())
		print('Save path:', args.save_path)
		if n_gpus == 1:
			main_worker(args)

