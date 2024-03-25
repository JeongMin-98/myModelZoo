import torch.utils.data
from .torch_model import DeepNetwork
from utils.dataLoader import download_mnist_data
from utils.tools import *
from network.exampleNet import *
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


def run_fn(args):
    device = check_device()
    model = MyCustomMnistModel(args)
    model.build_model(device)

    # summary(model.network, (64, 28, 28), device="cuda")

    model.train_model(device)


class MyCustomMnistModel(DeepNetwork):
    def __init__(self, args):
        # deepNetwork의 인자 다 받기
        super().__init__(args)
        # default -> mnist Model
        self.config_dir = args['config_dir']

        self.config_dir = os.path.join(self.config_dir, self.model_name)
        check_folder(self.config_dir)

        config_path = os.path.join(self.config_dir, self.model_name + ".cfg")
        self.cfg = parse_model_config(config_path)
        check_folder(config_path)

    def build_model(self, device):
        train_data, test_data = download_mnist_data()
        self.dataset_num = train_data.__len__()

        """ Load dataset"""
        # implement soon
        self.loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        self.dataset_iter = iter(self.loader)

        """ Network """
        self.network = Net(config=self.cfg).to(device)

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

    # def train_step(self, real_images, label, device=torch.device('cuda')):
    #     return super().train_step(real_images, label, device=device)
    #
    # def train_model(self, device):
    #     super().train_model(device)
    #
    # def torch_save(self, idx):
    #     super().torch_save(idx)



