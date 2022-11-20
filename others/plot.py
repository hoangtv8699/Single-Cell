from contrastive.utils import *

path = 'logs/28_09_2022 03_09_48.log'

classification_train_loss1, classification_val_loss1, classification_train_loss2, classification_val_loss2,\
    embed_train_loss, embed_test_loss, predict_train_loss1, predict_test_loss1, predict_train_loss2, predict_test_loss2 = read_logs(path)

plot_loss(classification_train_loss1, classification_val_loss1)
plot_loss(classification_train_loss2, classification_val_loss2)
plot_loss(embed_train_loss, embed_test_loss)
plot_loss(predict_train_loss1, predict_test_loss1)
plot_loss(predict_train_loss1, predict_test_loss1)


