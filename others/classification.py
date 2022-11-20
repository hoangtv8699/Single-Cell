import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from argparse import Namespace
from torch.utils.data import DataLoader

from datetime import datetime

from contrastive.utils import *

device = torch.device("cuda:0")

dataset_path = '../data/explore/multiome/'

param = {
    'input_train_mod': f'{dataset_path}multiome_gex_processed_training.h5ad',
    'output_pretrain': 'pretrain',
    'save_model_path': 'saved_model/',
    'logs_path': 'logs/'
}

args = Namespace(
    input_feats=13431,
    num_class=22,
    hid_feats=256,
    random_seed=17,
    activation='relu',
    num_layer=7,  # if using residual, this number must be odd
    epochs=1000,
    lr=1e-4,
    normalization='batch',
    patience=10
)

now = datetime.now()
logger = open(f'{param["logs_path"]}{now.strftime("%d_%m_%Y %H_%M_%S")}.log', 'w')
logger.write(str(args) + '\n')

# get feature type
logging.info('Reading `h5ad` files...')
train_mod = sc.read_h5ad(param['input_train_mod'])
mod = train_mod.var['feature_types'][0]

# get input and encode label
input_train = train_mod.layers["counts"]
LE = LabelEncoder()
train_mod.obs["class_label"] = LE.fit_transform(train_mod.obs["cell_type"])
input_label = train_mod.obs["class_label"].to_numpy()
logger.write('class name: ' + str(LE.classes_) + '\n')
args.classes_ = LE.classes_
args.num_class = len(args.classes_)

# 10-fold cross validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.random_seed)

i = 0
for train_index, val_index in skf.split(input_train, input_label):
    # get fold and train
    train_input = input_train[train_index]
    val_input = input_train[val_index]
    train_label = input_label[train_index]
    val_label = input_label[val_index]

    training_set = ModalityDataset(train_input, train_label, types='classification')
    val_set = ModalityDataset(val_input, val_label, types='classification')

    params = {'batch_size': 256,
              'shuffle': True,
              'num_workers': 0}

    train_loader = DataLoader(training_set, **params)
    val_loader = DataLoader(val_set, **params)

    net = CellClassification(args)
    net.cuda()
    logger.write(str(net) + '\n')
    opt = torch.optim.Adam(net.parameters(), args.lr)

    training_loss = []
    val_loss = []
    criterion = nn.CrossEntropyLoss()
    trigger_times = 0
    best_loss = 10000
    best_state_dict = net.state_dict()

    for epoch in range(args.epochs):
        logger.write(f'epoch:  {epoch}\n')

        # training
        net.train()
        running_loss = 0
        for train_batch, label in train_loader:
            train_batch, label = train_batch.cuda(), label.cuda()

            opt.zero_grad()
            out = net(train_batch, residual=True)
            loss = criterion(out, label)
            running_loss += loss.item() * train_batch.size(0)
            loss.backward()
            opt.step()

        training_loss.append(running_loss / len(train_loader.dataset))
        logger.write(f'training loss:  {training_loss[-1]}\n')
        logger.flush()

        # validating
        net.eval()
        running_loss = 0
        y_pred = []
        y_true = []
        with torch.no_grad():
            for val_batch, label in val_loader:
                val_batch, label = val_batch.cuda(), label.cuda()

                out = net(val_batch, residual=True)
                loss = criterion(out, label)
                running_loss += loss.item() * val_batch.size(0)
                if len(y_pred) == 0:
                    y_pred = out
                    y_true = label
                else:
                    y_pred = torch.cat((y_pred, out), dim=0)
                    y_true = torch.cat((y_true, label), dim=0)

            acc = cal_acc(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            val_loss.append(running_loss / len(val_loader.dataset))
            report = classification_report(y_true.detach().cpu().numpy(), np.argmax(y_pred.detach().cpu().numpy(), axis=-1),
                                           target_names=args.classes_, zero_division=1)
        logger.write(f'validation loss:  {val_loss[-1]}\n')
        logger.write(f'validation acc:  {acc}\n')
        logger.write(f'classification report:  \n{report}\n')
        logger.flush()

        # early stopping
        if len(val_loss) > 2 and val_loss[-1] >= best_loss:
            trigger_times += 1
            if trigger_times >= args.patience:
                logger.write(f'early stopping for mod trigger\n')
                logger.flush()
                break
        else:
            best_loss = val_loss[-1]
            best_state_dict = net.state_dict()
            trigger_times = 0

        print(epoch)
    i += 1
    torch.save(best_state_dict, f'{param["save_model_path"]}/10 fold/model param classification {i} {mod} {now.strftime("%d_%m_%Y %H_%M_%S")}.pkl')
