import os
import csv
import sys
import time

__all__ = ['CSVLogger', 'progress_bar']

class CSVLogger:
    def __init__(self, filename, headers, rounding=4, save=True):
        self.save = save
        if self.save:
            if not os.path.exists(filename):
                with open(filename, 'a+') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow(headers)
            self.filename = filename
            self.headers = headers
            self.rounding = rounding

    def save_arguments(self, where, arguments):
        if self.save:
            lines = ''.join(['{name} = {value}\n'.format(name=name, value=value) \
                for name, value in vars(arguments).items()])
            with open('{location}/parameters.txt'.format(location=where), 'w+') as f:
                f.write(lines)

    def __call__(self, row_data):
        if self.save:
            row = [row_data[key] for key in self.headers]
            for i in range(len(row)):
                if isinstance(row[i], float):
                    row[i] = round(row[i], self.rounding)
            with open(self.filename, 'a+') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(row)

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 40.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Total: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
