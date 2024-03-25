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
    if args['phase'] == "train":
        model.train_model(device)
    if args["phase"] == "test":
        model.test_model(device)


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
        train_data = download_mnist_data(train=True)
        self.dataset_num = train_data.__len__()

        """ Load dataset"""
        # implement soon
        self.loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
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

    def test_model(self, device):

        print()
        print("=======================================")
        print("phase : Test ")

        test_data = download_mnist_data(train=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        self.network.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.network(images)
                # _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += accuracy(outputs, labels)

        acc = 100 * correct / total
        print(self.acc_log_template.format(1, 1, acc))
        return acc
