__author__ = 'alex'

from flask import Flask, make_response, render_template
from flask.ext.script import Manager, Server
from log_loader import *
from progress_visulization import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure

import argparse

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('logs', type=str, nargs='+',
                    help="the path to log file")
parser.add_argument('--ip', dest="server_ip",
                    help='The ip of the server, set it to your actual ip to make it available to all ip')
parser.add_argument('--update', dest='refresh_interval',
                    default=10,
                    help='The interval between page refresh')
parser.add_argument('--msg', dest="msg", type=str, nargs='+',
                    default=[''],
                    help='add setup msgs')

args = parser.parse_args()


log_file_list = args.logs

setup_msg = ' '.join(args.msg)



class WebMonitor(object):
    def __init__(self, log_files, fig, training_loss_id=3, testing_loss_id=2):
        """
        Initialize and record log list
        :param log_files:
        :return:
        """
        self.log_files = log_files
        self.fig = fig
        self.axes = []
        self.training_loss_id=training_loss_id
        self.testing_loss_id = testing_loss_id

    def show_value(self, fd=None):
        """
        Internal function to read updated progress
        :return:
        """
        numbers = select_log_part(load_log(self.log_files),[('Testing','Testing', 8, 2), ('Training','loss', 4, 0)])
        y0, y1 = draw_loss(numbers, self.axes[0], self.training_loss_id, self.testing_loss_id)
        y2 = draw_acc(numbers, self.axes[1])
        return self.fig

    def render(self):
        for i in xrange(1,3):
            self.axes.append(self.fig.add_subplot(2,1,i, sharex=self.axes[0] if i>1 else None))

        return self.show_value()



@app.route('/get_image')
def draw_curve():
    fig = Figure(figsize=(20,10))
    monitor = WebMonitor(log_file_list, fig, training_loss_id=1, testing_loss_id=0)
    render_fig = monitor.render()
    canvas = FigureCanvas(render_fig)
    import StringIO
    png_out = StringIO.StringIO()
    canvas.print_png(png_out)
    response = make_response(png_out.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

@app.route('/')
def get_panel():
    return render_template('main_panel.html', server_ip='192.168.72.107', refresh_interval=args.refresh_interval,
                           setup=setup_msg)

if __name__ == "__main__":
    app.run(host=args.server_ip, port=10000, debug=True)

