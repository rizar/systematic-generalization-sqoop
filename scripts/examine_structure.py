import torch
import argparse
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import numpy as np
import h5py
from scipy.misc import imread, imresize, imsave

import glob
import vr.utils as utils
import vr.programs
from vr.data import ClevrDataset, ClevrDataLoader
from vr.preprocess import tokenize, encode


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=None)
parser.add_argument('--data_dir', default=None, type=str)

loss_fn = torch.nn.CrossEntropyLoss().to(device)

def run_our_model_batch(args, model, loader, dtype):
    model.type(dtype)
    model.eval()

    num_correct, num_samples = 0, 0
    total_loss = 0.0


    start = time.time()
    for batch in tqdm(loader):
        questions, images, feats, answers, programs, program_lists = batch
        if isinstance(questions, list):
            questions_var = Variable(questions[0].type(dtype).long(), volatile=True)
            q_types += [questions[1].cpu().numpy()]
        else:
            questions_var = Variable(questions.type(dtype).long(), volatile=True)
        feats_var = Variable(feats.type(dtype), volatile=True)
        answers_var = Variable(answers.to(device))
        scores = model(feats_var, questions_var)
        loss = loss_fn(scores, answers_var)
        probs = F.softmax(scores)
        total_loss += loss.data.cpu()
        _, preds = scores.data.cpu().max(1)
        num_correct += np.sum(preds == answers)
        num_samples += len(question)

    acc = float(num_correct) / num_samples
    print('Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))
    print('%.2fs to evaluate' % (start - time.time()))

    print(loss)




def main(args):
    all_checkpoints = glob.glob('%s/*.pt' %args.model_path)
    print(all_checkpoints)

    for i, checkpoint in enumerate(all_checkpoints):

        model, _ = utils.load_execution_engine(checkpoint, False, 'SHNMN')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

        f = open('f_%d.txt' %i, 'w')
        f.write('%s\n' %checkpoint)
        f.write('HARD_TAU | HARD_ALPHA \n')
        f.write('%s-%s\n'%(model.hard_code_tau, model.hard_code_alpha))
        f.write('TAUS\n')
        f.write('p(model) : %s\n' %str(F.sigmoid(model.model_bernoulli)))
        for i in range(3):
            tau0 = model.tau_0[i, :(i+2)] if model.hard_code_tau else F.softmax(model.tau_0[i, :(i+2)] )
            f.write('tau0: %s\n' %str(tau0.data.cpu().numpy()))
            tau1 = model.tau_1[i, :(i+2)] if model.hard_code_tau else F.softmax(model.tau_1[i, :(i+2)] )
            f.write('tau1: %s\n' %str(tau1.data.cpu().numpy()))

        f.write('ALPHAS\n')
        for i in range(3):
            alpha = model.alpha[i] if model.hard_code_alpha else F.softmax(model.alpha[i])
            f.write('alpha: %s\n' % " ".join(['{:.3f}'.format(float(x)) for x in alpha.view(-1).data.numpy()]))

        f.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

