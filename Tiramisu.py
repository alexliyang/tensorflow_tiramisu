import tensorflow as tf
import matplotlib
import tfplot
import numpy as np
slim=tf.contrib.slim

# tensorflow attempt to achieve
# https://arxiv.org/pdf/1611.09326.pdf

def figure_heatmap(data, cmap=matplotlib.cm.Set1):# Hier Farbe anpassbar
    fig, ax = tfplot.subplots()
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5,8.5,1), cmap.N)
    ax.imshow(data, cmap=cmap, norm=norm)
    return fig

class Tiramisu():
    #def __init__(self, edgelength=256, CLASSES=8, BANDS=3, k=16, layers=[4,5,7,10,12,15]):
    def __init__(self, edgelength=64, CLASSES=5, BANDS=4, k=48, layers=[4,5,7,10,12,15]): 
        
        lgts, lbls = self._build_tiramisu(edgelength, CLASSES, BANDS, k, layers)        
        
        flat_logits = tf.reshape(lgts, [-1, edgelength*edgelength, CLASSES])
        flat_labels = tf.reshape(lbls, [-1, edgelength*edgelength])
        
        flat_w = tf.where(flat_labels > 0, tf.ones_like(flat_labels), tf.zeros_like(flat_labels))
        
        
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")
        self.running_vars_initializer = tf.variables_initializer(var_list=running_vars)
                
        self._trainer(flat_logits, flat_labels, CLASSES, flat_w)        
        self._accuracy(flat_logits, flat_labels, flat_w)       
        self.merged = tf.summary.merge_all()
        
    def _Layer(self, inp, k, phase, kp):
        inp = tf.layers.batch_normalization(inp, training=phase)
        inp = tf.nn.relu(inp)        
        inp = slim.conv2d(inp, k, 3, 1, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))
        return slim.dropout(inp, kp)
    
    def _DenseBlock(self, inp, phase, l, k, kp):
        layers = []
        for i in range(l):
            layer = self._Layer(inp, k, phase, kp)
            layers.append(layer)
            inp = tf.concat([inp, layer], axis=3)
        return tf.concat(layers, axis=3)        
        
    def _TransitionDown(self, inp, phase, kp):        
        inp = tf.layers.batch_normalization(inp, training=phase)
        inp = tf.nn.relu(inp)
        inp = slim.conv2d(inp, inp.shape[3], 1, 1, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))
        inp = slim.dropout(inp, kp)
        inp = slim.max_pool2d(inp, (2, 2))
        return inp
        
    def _TransitionUp(self, inp, filters):
        inp = slim.conv2d_transpose(inp, filters, kernel_size=3, stride=2, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))
        return inp
    
    def _build_tiramisu(self, edgelength, CLASSES, BANDS, k, layers):
        self.X = tf.placeholder('float', shape=[None, edgelength, edgelength, BANDS])
        tf.summary.image('input_x', self.X[:3, :, :, :3], 3)        
        
        self.y = tf.placeholder('int32', shape=[None, edgelength, edgelength, 1])
        self.ysub = tf.stack([self.y[0],self.y[1],self.y[2]], axis=0)
        
        summary_heatmap = tfplot.summary.wrap(figure_heatmap, batch=True)
        summary_heatmap("input_y", self.ysub[:, :, :, 0])
        
        self.phase = tf.placeholder('bool')
        self.kp = tf.placeholder('float')
        
        concatenations = []
        
        cycle_start = slim.conv2d(self.X, 48, kernel_size=3, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32), scope='conv_init')        
                
        for l in layers[:-1]:
            blocked = self._DenseBlock(cycle_start, self.phase, l, k, self.kp)
            concatenated = tf.concat([cycle_start, blocked], axis=3)
            concatenations.append(concatenated)
            cycle_start = self._TransitionDown(concatenated, self.phase, self.kp)
        
        blocked = self._DenseBlock(cycle_start, self.phase, layers[-1], k, self.kp)        
        
        for i, l in enumerate(layers[::-1][:-1]):
            transitioned_up = self._TransitionUp(blocked, l*k + layers[len(layers)-i-2]*k)
            concatenated = tf.concat([concatenations[len(layers)-2-i],transitioned_up], axis=3)
            blocked = self._DenseBlock(concatenated, self.phase, l, k, self.kp)
            
        conv_final = slim.conv2d(blocked, CLASSES, 3, 1, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))
        conv_finalsub = tf.cast(tf.argmax(tf.stack([conv_final[0],conv_final[1],conv_final[2]], 0), 3), tf.uint8)
        print(conv_finalsub)
        summary_heatmap("output_pred", conv_finalsub)
        
        return conv_final, self.y
        
    def _trainer(self, logits, labels, n_classes, w):
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=w))
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = tf.train.AdamOptimizer(1e-4, .995).minimize(self.loss)#1e-3            
        tf.summary.scalar('loss', self.loss)
            
    def _accuracy(self, logits, labels, w):
        predictions = tf.argmax(logits, 2)
        self.acc = tf.metrics.accuracy(labels, predictions, w, name="my_metric")
        tf.summary.scalar('accuracy', self.acc[1])