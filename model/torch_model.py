import torch.utils.data
from torch.nn.functional import one_hot
from utils.dataLoader import ImageDataset
from utils.tools import *
from network.exampleNet import *
from torch.utils.tensorboard import SummaryWriter

from torchsummary import summary


def run_fn(args):
    device = check_device()
    model = DeepNetwork(args)
    model.build_model(device)
    model.train_model(device)


class DeepNetwork():
    def __init__(self, args):
        super(DeepNetwork, self).__init__()
        self.model_name = args['model_name']
        self.checkpoint_dir = args['checkpoint_dir']
        self.result_dir = args['result_dir']
        self.log_dir = args['log_dir']
        self.sample_dir = args['sample_dir']
        self.config_dir = args['config_dir']
        self.dataset_name = args['dataset']

        """ Network parameters """
        self.feature_size = args['feature_size']

        """ Training parameters """
        self.lr = args['lr']
        self.iteration = args['iteration']
        self.img_size = args['img_size']
        self.batch_size = args['batch_size']
        # FIX global_batch_size
        self.global_batch_size = self.batch_size

        """ Misc """
        # self.save_freq = args['save_freq']
        self.log_template = 'step [{}/{}]: loss: {:.3f}'
        self.acc_log_template = 'step [{}/{}]: accuracy: {:.3f}'

        """ Directory """
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        check_folder(self.checkpoint_dir)
        self.log_dir = os.path.join(self.log_dir, self.model_dir)
        check_folder(self.log_dir)

        """ Dataset """
        dataset_path = './dataset'
        self.dataset_path = os.path.join(dataset_path, self.dataset_name)

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self, device):
        """ Dataset Load """
        dataset = ImageDataset(dataset_path=self.dataset_path, img_size=self.img_size)
        self.dataset_num = dataset.__len__()
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=1, shuffle=True)
        self.dataset_iter = iter(self.loader)

        """ Network """
        self.network = Net(input_shape=self.img_size, feature_size=self.feature_size).to(device)

        """ Optimizer """
        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        """ Checkpoint """
        latest_ckpt_name, start_iter = find_latest_ckpt(self.checkpoint_dir)

        if latest_ckpt_name is not None:
            # "rank == 0" means a first gpu.
            print('Latest checkpoint restored!! ', latest_ckpt_name)
            print('start iteration : ', start_iter)
            self.start_iteration = start_iter

            latest_ckpt = os.path.join(self.checkpoint_dir, latest_ckpt_name)
            ckpt = torch.load(latest_ckpt, map_location=device)

            self.network.load_state_dict(ckpt["network"])
            self.optim.load_state_dict(ckpt["optim"])
        else:
            self.start_iteration = 0

    def train_step(self, real_images, label, device=torch.device('cuda')):
        # gradient check
        requires_grad(self.network, True)

        # forward pass
        logit = self.network(real_images)
        # loss
        loss = cross_entroy_loss(logit, label)

        # backword
        apply_gradients(loss, self.optim)

        return logit, loss

    def train_model(self, device):
        # setup tensorboard
        train_summary_writer = SummaryWriter(self.log_dir)

        # start training
        print()
        print(self.dataset_path)
        print("Dataset number : ", self.dataset_num)
        print("Each batch size : ", self.batch_size)
        print("Global batch size : ", self.global_batch_size)
        print("Target image size : ", self.img_size)
        # print("Save frequency : ", self.save_freq)
        print("PyTorch Version :", torch.__version__)
        print('max_steps: {}'.format(self.iteration))
        print()

        train_loss_list = []

        print("=======================================")
        print("Phase : training")

        best_loss = 1e9

        # number of self.dataset_iter
        iter_per_epoch = max(self.dataset_num // self.batch_size, 1)
        epoch = 0
        acc = 0
        for idx in range(self.start_iteration, self.iteration):

            if idx == 0:
                print("=======================================")
                print("count params")
                n_params = count_parameters(self.network)
                print("network parameters : ", format(n_params, ','))
                print("=======================================")

            if idx % iter_per_epoch == 0:
                if idx == 0:
                    continue
                self.dataset_iter = iter(self.loader)
                """ calculate average loss for each epoch """
                loss_per_epoch = sum(train_loss_list) / iter_per_epoch

                print("=======================================")
                print("epoch")
                print(self.log_template.format(epoch, self.iteration // iter_per_epoch, loss_per_epoch))
                print()

                print("Accuracy")
                print(self.acc_log_template.format(epoch, self.iteration // iter_per_epoch, acc / iter_per_epoch * 100))
                print("=======================================")
                """ Each epoch, Save model """
                self.torch_save(idx)

                loss = 0
                train_loss_list = []
                epoch += 1

            real_img, label = next(self.dataset_iter)
            real_img = real_img.to(device)
            label = label.to(device)

            logit, loss = self.train_step(real_img, label, device=device)
            acc = accuracy(logit, label)
            train_loss_list.append(loss)
            train_summary_writer.add_scalar('acc', acc, global_step=idx)
            train_summary_writer.add_scalar('loss', loss, global_step=idx)

        print("=======================================")

    def torch_save(self, idx):
        print()
        print("Saving model")
        print("path : " + os.path.join(self.checkpoint_dir, "iter_{}.pt".format(idx)))
        torch.save(
            {
                'network': self.network.state_dict(),
                'optim': self.optim.state_dict()
            },
            os.path.join(self.checkpoint_dir, 'iter_{}.pt'.format(idx))
        )

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.img_size)
