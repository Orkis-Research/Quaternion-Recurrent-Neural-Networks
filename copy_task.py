import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Module
from torch import nn
from torch.nn.init import orthogonal
from complexops import get_modulus
from complex_models import CRN, MultiLayerCRN
from complex_models import ComplexLinear
from model import RNNModel
import argparse, os
import shutil
import time
import math


def copy_task_gen(batch_size, T, rng, length_seq=10):
    blank = np.zeros((batch_size, T - 1))
    seq = rng.randint(low=1, high=9, size=(batch_size, length_seq))
    delimiter = np.ones((batch_size, 1)) * 9
    blank2 = np.zeros((batch_size, length_seq))
    x = np.concatenate([seq, blank, delimiter, blank2], axis=1)
    zeros2 = np.zeros((batch_size, T))
    y = np.concatenate([blank2, zeros2, seq[:]], axis=1)
    yield x.T ,y.T


def get_batch(batch_size, T, rng, length_seq=10, evaluation=False):
    use_cuda = torch.cuda.is_available()
    data, target = next(copy_task_gen(batch_size, T, rng, length_seq))
    data   = Variable(torch.LongTensor(data),   volatile=evaluation)
    target = Variable(torch.LongTensor(target), volatile=evaluation)
    if use_cuda:
        data   = data.cuda()
        target = target.cuda()
    return data, target


def evaluate_batch(model, criterion, T, rng, ntokens=10, length_seq=10, batch_size=80):
    use_cuda = torch.cuda.is_available()
    # Turn on evaluation mode which disables dropout.
    model.eval()
    # See http://pytorch.org/docs/0.3.0/nn.html?highlight=eval#torch.nn.Module.eval
    # it says: "Sets the module in evaluation mode. This has any effect only
    #           on modules such as Dropout or BatchNorm."
    hidden = model.init_hidden(batch_size)
    if model.rnn_type not in {'LSTM', 'CLSTM', 'QLSTM'}:
        hidden = hidden.cuda() if use_cuda else hidden
    else:
        h0, c0 = hidden
        h0 = h0.cuda() if use_cuda else h0
        c0 = c0.cuda() if use_cuda else c0
        hidden = (h0, c0)
    data, target = get_batch(batch_size, T, rng, length_seq, evaluation=True)
    output, hidden = model(data, hidden)
    output_flat = output.view(-1, ntokens)
    loss = criterion(output_flat, target.view(target.size(0) * target.size(1))).data
    return loss[0]


def train(model, optimizer, criterion, T, rng, ntokens=10, length_seq=10, batch_size=80, clip=1,
          nb_batch_iters=1000, nb_valid_iters=1000, print_interval=10, evaluation_interval=10,
          start_iter=1, best_iter=0, pplbest=np.inf, training_losses=[], ppl_scores=[],
          save_path="./"):
    use_cuda = torch.cuda.is_available()
    # Turn on training mode which enables dropout.
    model.train()
    # See: http://pytorch.org/docs/0.3.0/nn.html?highlight=train#torch.nn.Module.train
    # it says: "Sets the module in training mode. This has any effect only
    #           on modules such as Dropout or BatchNorm."
    total_loss = 0
    start_time = time.time()
    for i in range(start_iter, nb_batch_iters + 1):
        data, target = get_batch(batch_size, T, rng, length_seq, evaluation=False)
        optimizer.zero_grad()
        hidden = model.init_hidden(batch_size)
        if model.rnn_type not in {'LSTM', 'CLSTM', 'QLSTM'}:
            hidden = hidden.cuda() if use_cuda else hidden
        else:  # which means if rnn_type == 'LSTM'
            if type(hidden) is tuple:
                # which means if hidden nb_layers == 1:
                # (an element could be a tuple of 2 tuples when we have bidirectional layers)
                # because otherwise we use a list when we have more than one layer. 
                h0, c0 = hidden
                if type(h0) is not tuple:  # in case it is 1-directional
                    h0 = h0.cuda() if use_cuda else h0
                    c0 = c0.cuda() if use_cuda else c0
                    hidden = (h0, c0)
                else:                      # in case it is 2-directional
                    h0 = (h0[0].cuda(), h0[1].cuda()) if use_cuda else h0
                    c0 = (c0[0].cuda(), c0[1].cuda()) if use_cuda else c0
                    hidden = (h0, c0)
            else:
                # which mean we have multilayers. hidden here is a list of tuples
                new_hidden = []
                if type(hidden[0][0]) is not tuple: # which means if it is 1-directional
                    for j in range(len(hidden)):
                        h0, c0 = hidden[j]
                        h0 = h0.cuda() if use_cuda else h0
                        c0 = c0.cuda() if use_cuda else c0
                        t = (h0, c0)
                        new_hidden.append(t)
                    hidden = new_hidden
                else:                               # which mean if it is 2-directional
                    for j in range(len(hidden)):
                        h0, c0 = hidden[j][0]
                        h1, c1 = hidden[j][1]
                        h0 = h0.cuda() if use_cuda else h0
                        c0 = c0.cuda() if use_cuda else c0
                        h1 = h1.cuda() if use_cuda else h1
                        c1 = c1.cuda() if use_cuda else c1                    
                        t = ((h0, c0), (h1, c1))
                        new_hidden.append(t)
                    hidden = new_hidden
        output, hidden = model(data, hidden)
        output = output.contiguous()
        target = target.contiguous()
        loss = criterion(output.view(-1, ntokens), target.view(target.size(0) * target.size(1)))
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm(model.parameters(), max_norm=clip)
        optimizer.step()
        total_loss += loss.data
        # printing the loss
        if i % print_interval == 0:
            cur_loss = total_loss[0] / print_interval
            elapsed = time.time() - start_time
            print('| iteration {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  'loss {:5.6f} | ppl {:8.6f}'.format(
                      i, i, nb_batch_iters, elapsed * 1000 / print_interval,
                      cur_loss, math.exp(cur_loss)
                  )
            )
            training_losses.append(cur_loss)
            ppl_scores.append(math.exp(cur_loss))
            total_loss = 0
            start_time = time.time()
        if i % evaluation_interval == 0:
            pplscore = math.exp(cur_loss)
            if pplscore < pplbest:
                is_best = True
                pplbest = pplscore
                best_iter = i
            else:
                is_best = False
            state = {
                'iter'              :      i,
                'rnn_state_dict'    :      model.state_dict(),
                'pplscore'          :      pplscore,
                'pplbest'           :      pplbest,
                'optimizer'         :      optimizer.state_dict(),
                'clip'              :      clip,
                'nb_batch_iters'    :      nb_batch_iters,
                'nb_valid_iters'    :      nb_valid_iters,
                'print_interval'    :      print_interval,
                'T'                 :      T,
                'length_seq'        :      length_seq,
                'eval_interval'     :      evaluation_interval,
                'batch_size'        :      batch_size,
                'rng'               :      rng,
                'best_iter'         :      best_iter,
                'training_losses'   :      training_losses,
                'ppl_scores'        :      ppl_scores, 
            }
            if is_best:
                print("\rSaving Best Model.\r")
                save_checkpoint(state, is_best, save_path=save_path)
                save_checkpoint(state, is_best=False, save_path=save_path)
            else:
                print("\rSaving last Model.\r")
                save_checkpoint(state, is_best, save_path=save_path)


    # now after training we perform evaluation
    valid_loss = 0
    start_time = time.time()
    print("Loading Best model for validation:")
    bestpoint  = torch.load(os.path.join(save_path + '/model_best.pth.tar'))
    model.load_state_dict(bestpoint['rnn_state_dict'])
    for i in range(nb_valid_iters):
        valid_loss += evaluate_batch(model, criterion, T, rng, ntokens, length_seq, batch_size)
        if (i + 1) % print_interval == 0:
            cur_loss = valid_loss / (i + 1)
            elapsed = time.time() - start_time
            print('| iteration {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  'loss {:5.6f} | ppl {:8.6f}'.format(
                      i + 1, i + 1, nb_valid_iters, elapsed * 1000 / print_interval,
                      cur_loss, math.exp(cur_loss)
                  )
            )
            start_time = time.time()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, is_best, save_path='./', filename='checkpoint.pth.tar'):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_path, 'model_best.pth.tar'))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Argument Parser for RNN model.")

    parser.add_argument(
        "-w", 
        "--workdir", 
        default=".", 
        type=str,
        help="Path to the workspace directory for this experiment."
    )

    parser.add_argument(
        # CLSTM and CGRU are respectively Complex LSTM and Complex GRU.
        '--rnn-type',
        choices=['LSTM', 'GRU', 'CLSTM', 'CGRU', 'CRU', 'RNN_TANH', 'RNN_RELU', 'QLSTM', 'QGRU', 'QRNN'],
        default='CRU',
        type=str,
        help='The type of the RNN cell.'
    )
    parser.add_argument(
        # This is going to be the hidden size as well the embedding size
        '--hidden-size',
        default=128,
        type=int,
        help='This is going to be the hidden size as well the embedding size.'
    )
    parser.add_argument(
        '--nb-layers',
        default=1,
        type=int,
        help='Number of layers in the RNN.'
    )
    parser.add_argument(
        '--seed',
        default=1337, 
        type=int,
        help="Seed for PRNGs."
    )
    parser.add_argument(
        '--dropout',
        default=0, 
        type=float,
        help="RNN dropout probability."
    )
    parser.add_argument(
        '--dropout-type',
        choices=['regular', 'complex'],
        default='regular',
        type=str,
        help='The type of dropout to perform.',
    )
    parser.add_argument(
        '--bidirectional',
        default=False,
        type=bool,
        help="Using a bidirectional RNN or a 1-directional RNN."
    )
    parser.add_argument(
        '--operation',
        choices=['convolution', 'linear'],
        default='convolution',
        type=str,
        help="Using a convolutional or a fully connected RNN."
    )
    parser.add_argument(
        '--clip',
        default=1,
        type=float,
        help="Gradient norm above which we perform the clipping."
    )
    parser.add_argument(
        '--batch-size',
        default=80,
        type=int,
        help="Batch Size."
    )
    parser.add_argument(
        '--print-interval',
        default=10,
        type=int,
        help="Number of training iterations after which we print the training loss."
    )
    parser.add_argument(
        '--eval-interval',
        default=10,
        type=int,
        help="Number of training iterations after which we check if the model is yielding the lowest training loss. \
              In such case, it is saved"
    )
    parser.add_argument(
        '--batch-iters',
        default=10000,
        type=int,
        help="Number of batch iterations to perform (you can consider it as something related to nb of epochs)."
    )
    parser.add_argument(
        '--valid-iters',
        default=1000,
        type=int,
        help="Number of iterations between which the evalution of the model on the validation set is performed."
    )
    parser.add_argument(
        '--lr',
        default=0.01,
        type=float,
        help="Learning rate."
    )
    parser.add_argument(
        '--in-channels',
        default=1,
        type=int,
        help="Input feature maps."
    )
    parser.add_argument(
        '--out-channels',
        default=4,
        type=int,
        help="Output feature maps."
    )
    parser.add_argument(
        '--kernel-size',
        default=3,
        type=int,
        help="Kernel size."
    )
    parser.add_argument(
        '--rec-kernel-size',
        default=3,
        type=int,
        help="Recurrent Kernel size."
    )
    parser.add_argument(
        '--stride',
        default=1,
        type=int,
        help="Stride size."
    )
    parser.add_argument(
        '--padding',
        default=1,
        type=int,
        help="Padding size."
    )
    parser.add_argument(
        '--dilation',
        default=1,
        type=int,
        help="Dilation size."
    )
    parser.add_argument(
        # weight intialization would be orthogonal in case when the RNN is real-valued.
        '--weight-init',
        choices=['unitary', 'complex', 'orthogonal', 'quaternion'],
        default='unitary',
        type=str,
        help="Weight initialization."
    )
    parser.add_argument(
        '--w-mod-init',
        default='orthogonal',
        type=str,
        help="Weight initialization for the modulation gate."
    )
    parser.add_argument(
        '--modact',
        default='softmax',
        type=str,
        help="Activation for the modulation gate."
    )
    parser.add_argument(
        '--tie-weights',
        default=False,
        type=bool,
        help="Either using the embedding weights for the classification or not."
    )
    parser.add_argument(
        '--T',
        default=100,
        type=int,
        help="Number of time steps that consists of having blanks."
    )
    parser.add_argument(
        '--length-seq',
        default=10,
        type=int,
        help="Length of the sequence to memorize."
    )
    parser.add_argument(
        '--resume-path',
        default='',
        type=str,
        help='path to latest checkpoint (default: none)')
    parser.add_argument(
        '--save-path',
        default="./",
        type=str,
        help='The directory where to save the checkpoints.'
    )

    return parser


def parse_args(argv):
    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv
    return opt


def main(argv=None):
    opt = parse_args(argv)

    rnn_type                =               opt.rnn_type
    hidden_size             =               opt.hidden_size
    nlayers                 =               opt.nb_layers
    seed                    =               opt.seed
    dropout                 =               opt.dropout
    dropout_type            =               opt.dropout_type
    bidirectional           =               opt.bidirectional
    operation               =               opt.operation
    clip                    =               opt.clip
    batch_size              =               opt.batch_size
    print_interval          =               opt.print_interval
    eval_interval           =               opt.eval_interval
    nb_batch_iters          =               opt.batch_iters
    nb_valid_iters          =               opt.valid_iters
    lr                      =               opt.lr
    in_channels             =               opt.in_channels
    out_channels            =               opt.out_channels
    kernel_size             =               opt.kernel_size
    rec_kernel_size         =               opt.rec_kernel_size
    stride                  =               opt.stride
    padding                 =               opt.padding
    dilation                =               opt.dilation
    weight_init             =               opt.weight_init
    w_mod_init              =               opt.w_mod_init
    modact                  =               opt.modact
    tie_weights             =               opt.tie_weights
    T                       =               opt.T
    length_seq              =               opt.length_seq
    
    resume_path             =               opt.resume_path
    save_path               =               opt.save_path
    workdir                 =               opt.workdir

    if resume_path:
        resume_path         =               os.path.join(workdir, resume_path)
    save_path               =               workdir # os.path.join(workdir, save_path)
    chkptFilename = os.path.join(workdir, 'checkpoint.pth.tar')
    isResuming    = os.path.isfile(chkptFilename) or resume_path
    if isResuming and (not resume_path):
        resume_path = chkptFilename


    if not os.path.isdir(workdir):
            os.mkdir(workdir)

    use_cuda = torch.cuda.is_available()
    if operation == 'linear':
        toy_model = RNNModel(
            rnn_type=rnn_type, ninp=hidden_size, nhid=hidden_size, nlayers=nlayers,
            ntoken=10, dropout=dropout, dropout_type=dropout_type, tie_weights=tie_weights,
            seed=seed, bidirectional=bidirectional, weight_init=weight_init, w_mod_init=w_mod_init,
            modulation_activation=modact
        )
    else:  # which means if operation is convolutional
        toy_model = RNNModel(
            rnn_type=rnn_type, ninp=hidden_size, nhid=None, nlayers=nlayers,
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            rec_kernel_size=rec_kernel_size, stride=stride, dilation=dilation, padding=padding,
            ntoken=10, dropout=dropout, dropout_type=dropout_type,
            bidirectional=bidirectional, tie_weights=tie_weights, seed=seed,
            weight_init=weight_init, w_mod_init=w_mod_init,
            modulation_activation=modact
        )

    toy_model = toy_model.cuda() if use_cuda else toy_model
    print(toy_model)
    print("Number of learnable parameters in the model " + str(count_parameters(toy_model)))
    optimizer = torch.optim.RMSprop(
        toy_model.parameters(), lr=lr, alpha=0.99, eps=1e-08,
        weight_decay=0.0001, momentum=0, centered=True)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.RandomState(seed)

    start_iter                = 0
    pplbest                   = np.inf
    best_iter                 = 0
    training_losses           = []
    ppl_scores                = []

    if resume_path:
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            for k in checkpoint.keys():
                if k not in ['rnn_state_dict',
                             'optimizer',
                             'training_losses',
                             'ppl_scores']:
                    print("    " + str(k) + \
                          ": " + str(checkpoint[k]))
            print('\n')
            print('-' * 100)

            start_iter                        =               checkpoint['iter']
            clip                              =               checkpoint['clip']
            nb_batch_iters                    =               checkpoint['nb_batch_iters']
            nb_valid_iters                    =               checkpoint['nb_valid_iters']
            batch_size                        =               checkpoint['batch_size']
            print_interval                    =               checkpoint['print_interval']
            eval_interval                     =               checkpoint['eval_interval']
            T                                 =               checkpoint['T']
            length_seq                        =               checkpoint['length_seq']
            pplscore                          =               checkpoint['pplscore']
            pplbest                           =               checkpoint['pplbest']
            best_iter                         =               checkpoint['best_iter']
            training_losses                   =               checkpoint['training_losses']
            ppl_scores                        =               checkpoint['ppl_scores']
            rng                               =               checkpoint['rng']

            toy_model.load_state_dict(checkpoint['rnn_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("    rnn_type:           "          + str(toy_model.rnn_type))
    print("    input_size:         "          + str(toy_model.ninp))
    print("    hidden_size:        "          + str(toy_model.nhid))
    print("    nb_layers:          "          + str(toy_model.nlayers))
    print("    operation:          "          + str(toy_model.operation))
    print("    bidirectional:      "          + str(toy_model.bidirectional))
    print("    dropout             "          + str(toy_model.dropout))
    print("    bias                "          + str(toy_model.bias))
    print("    tie_weights         "          + str(toy_model.tie_weights))
    print("    seed                "          + str(toy_model.seed))
    if toy_model.rnn_type in {'CGRU', 'CLSTM', 'CRU'}:
        print("    dropout_type             " + str(toy_model.dropout_type))
        print("    weight_init              " + str(toy_model.weight_init))
        if toy_model.rnn_type == 'CRU':
            print("    w_mod_init               " + str(toy_model.w_mod_init))
            print("    modulation_activation    " + str(toy_model.modulation_activation))
    if toy_model.operation == 'convolution':
        print("    in_channels:        "      + str(toy_model.in_channels))
        print("    out_channels:       "      + str(toy_model.out_channels))
        print("    kernel_size:        "      + str(toy_model.kernel_size))
        print("    rec_kernel_size:    "      + str(toy_model.rec_kernel_size))
        print("    stride:             "      + str(toy_model.stride))
        print("    padding:            "      + str(toy_model.padding))
        print("    dilation:           "      + str(toy_model.dilation))

    train(toy_model, optimizer, criterion, T=T, rng=rng, ntokens=10,
          length_seq=length_seq, batch_size=batch_size, clip=clip,
          nb_batch_iters=nb_batch_iters, nb_valid_iters=nb_valid_iters,
          print_interval=print_interval, evaluation_interval=print_interval,  # eval_interval,
          start_iter=start_iter+1, best_iter=best_iter, pplbest=pplbest,
          training_losses=training_losses, ppl_scores=ppl_scores,
          save_path=save_path)


if __name__ == "__main__":
    main()
