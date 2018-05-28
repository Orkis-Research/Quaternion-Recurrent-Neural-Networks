from __future__          import division
import sys
import torch
import torch.nn          as nn
from torch.nn            import Parameter
from torch.nn            import functional as F
import torch.optim
from torch.autograd      import Variable
import numpy             as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nn_models           import QRNN, RNN

#
# PLOT Functions
#
def getPlotArray(pred, inp):

    pred = pred.reshape(pred.shape[1], pred.shape[0], pred.shape[2])
    inp = inp.reshape(inp.shape[1], inp.shape[0], inp.shape[2])
    return (np.swapaxes(pred[0],1,0)),(np.swapaxes(inp[0],1,0))

#
# Convert to torch.Variable 
#
def tovar(x):
    return Variable(torch.FloatTensor(x).cuda(), requires_grad = False)

def getTask(N_BATCH, SEQ_LENGTH, FEAT_SIZE):
    data = []
    seq  = []
    for i in range(N_BATCH):
        for j in range(SEQ_LENGTH):
            feat = np.random.randint(2, size=FEAT_SIZE)
            seq.append(feat)
        data.append(seq)
        seq = []
    return np.array(data)

#
# DEFINING THE TASK
#

N_BATCH_TRAIN    = 2500
N_BATCH_TEST     = 1
SEQ_LENGTH       = 20
FEAT_SIZE        = 8
EPOCHS           = 15000
RNN_HIDDEN_SIZE  = 512
QRNN_HIDDEN_SIZE = 1024
RNN_NB_HIDDEN    = 1
QRNN_NB_HIDDEN   = 1

train = getTask(N_BATCH_TRAIN,SEQ_LENGTH, FEAT_SIZE)
train = train.reshape((SEQ_LENGTH,N_BATCH_TRAIN,FEAT_SIZE))
test  = getTask(N_BATCH_TEST,SEQ_LENGTH, FEAT_SIZE)
test  = test.reshape((SEQ_LENGTH,N_BATCH_TEST,FEAT_SIZE))

losses_r      = []
losses_q      = []
accs_r        = []
accs_q        = []
accs_test     = []

net_r = RNN(FEAT_SIZE, RNN_HIDDEN_SIZE, RNN_NB_HIDDEN).cuda()
net_q = QRNN(FEAT_SIZE, QRNN_HIDDEN_SIZE, QRNN_NB_HIDDEN).cuda()

nb_param_q = sum(p.numel() for p in net_q.parameters() if p.requires_grad)
nb_param_r = sum(p.numel() for p in net_r.parameters() if p.requires_grad)

print("QRNN & RNN Copy Task - Titouan Parcollet - LIA, ORKIS")
print("Models Infos --------------------")
print("(QRNN) Number of trainable parameters : "+str(nb_param_q))
print("(RNN)  Number of trainable parameters : "+str(nb_param_r))


#
# TRAINING LOOP
#

break_r = False
break_q = False
for epoch in range(EPOCHS):

    # RNN Training
    if break_r == False:
        net_r.zero_grad()
        p = net_r.forward(tovar(train))
        loss = nn.BCELoss()
        val_loss = loss(p, tovar(train))
        val_loss.backward()
        net_r.adam.step()
        
        # Train ACC and LOSS
        p = p.cpu().data.numpy()
        p[p>0.5]=1
        p[p<0.5]=0
        acc = np.sum(p == train) / train.size
        accs_r.append(acc)
        if (epoch % 5) == 0:
            losses_r.append(float(val_loss.data))
        if acc>0.995:
            break_r = True
        if (epoch % 10) == 0:
            string = "(RNN) It : "+str(epoch)+" | Train Loss = "+str(float(val_loss.data))+" | Train Acc = "+str(acc)
            print(string)

    # QRNN Training
    if break_q == False:
        net_q.zero_grad()
        p = net_q.forward(tovar(train))
        loss = nn.BCELoss()
        val_loss = loss(p, tovar(train))
        val_loss.backward()
        net_q.adam.step()
        p = p.cpu().data.numpy()
        p[p>0.5]=1
        p[p<0.5]=0

        acc = np.sum(p == train) / train.size
        accs_q.append(acc)
        if (epoch % 5) == 0:
            losses_q.append(float(val_loss.data))
        if acc>0.995:
            break_q = True
        if (epoch % 10) == 0:
            string = "(QRNN) It : "+str(epoch)+" | Train Loss = "+str(float(val_loss.data))+" | Train Acc = "+str(acc)
            print(string)
            
    # IF QRNN & RNN Training are done, end.
    if break_q == True and break_r == True:
        break

print("Training Ended - Saving Plots")

#
# Loss Curves
#
plt.plot(np.asarray(losses_q), 'b', label="QRNN")
plt.plot(np.asarray(losses_r), 'r', label="RNN")
plt.legend(loc='upper right')
plt.ylabel('Training Loss')
plt.xlabel('Epochs')
plt.savefig("curves.pdf", dpi=1500, bbox_inches='tight')

#
# Sequence Matrix Plot
#
p_rnn   = net_r.forward(tovar(test)).cpu().data.numpy()
p_qrnn  = net_q.forward(tovar(test)).cpu().data.numpy()
out_rnn, inp_rnn    = getPlotArray(p_rnn, test)
out_qrnn, inp_qrnn  = getPlotArray(p_qrnn, test)
p_rnn[p_rnn>0.5]=1
p_rnn[p_rnn<0.5]=0
acc_r = np.sum(p_rnn == test) / test.size
p_qrnn[p_rnn>0.5]=1
p_qrnn[p_rnn<0.5]=0
acc_q = np.sum(p_qrnn == test) / test.size
print("Final Results -------------------")
print("Acc for RNN : "+str(acc_r))
print("Acc for QRNN : "+str(acc_q))

f, axarr = plt.subplots(2,2, sharex=True)
plt.rcParams['image.cmap'] = 'Blues'
axarr[0,1].matshow(inp_qrnn)
axarr[0,1].axes.get_xaxis().set_ticks([])
axarr[0,1].axes.get_yaxis().set_ticks([])
axarr[1,1].matshow(out_qrnn)
axarr[1,1].axes.get_xaxis().set_ticks([])
axarr[1,1].axes.get_yaxis().set_ticks([])
im  = axarr[0,0].matshow(inp_rnn)
axarr[0,0].axes.get_xaxis().set_ticks([])
axarr[0,0].axes.get_yaxis().set_ticks([])
axarr[1,0].matshow(out_rnn)
axarr[1,0].axes.get_xaxis().set_ticks([])
axarr[1,0].axes.get_yaxis().set_ticks([])
axarr[0,0].set_title('RNN')
axarr[0,1].set_title('QRNN')
i=0
for ax in axarr.flat:
    if i==0:
        ax.set(ylabel='Inputs')
    elif i==2:
        ax.set(ylabel='Outputs')
    i+=1

plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
cbar_ax = f.add_axes([1.05, 0.15, 0.05, 0.7])
plt.colorbar(im, cax=cbar_ax)
plt.rcParams.update(params)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=-0.25)
plt.savefig("copy_res_test.pdf", dpi=1500, bbox_inches='tight')

print("Done ! That's All Folks ;) !")
#plt.show()