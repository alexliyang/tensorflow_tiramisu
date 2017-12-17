import tensorflow as tf
slim=tf.contrib.slim

# tensorflow attempt to achieve
# https://arxiv.org/pdf/1611.09326.pdf

class Tiramisu():
    def __init__(self, edgelength=256, CLASSES=2, BANDS=3, k=16, layers=[4,5,7,10,12,15]):        
        lgts, lbls = self._build_tiramisu(edgelength, CLASSES, BANDS, k, layers)
        self._trainer(lgts, lbls, CLASSES)
        
    def _Layer(self, inp, k, phase):
        inp = tf.layers.batch_normalization(inp, training=phase)
        inp = tf.nn.relu(inp)        
        inp = slim.conv2d(inp, k, 3, 1, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))
        return slim.dropout(inp, .8)
    
    def _DenseBlock(self, inp, phase, l, k):
        layers = []
        for i in range(l):
            layer = self._Layer(inp, k, phase)
            layers.append(layer)
            inp = tf.concat([inp, layer], axis=3)
        return tf.concat(layers, axis=3)        
        
    def _TransitionDown(self, inp, phase):        
        inp = tf.layers.batch_normalization(inp, training=phase)
        inp = tf.nn.relu(inp)
        inp = slim.conv2d(inp, inp.shape[3], 1, 1, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))
        inp = slim.dropout(inp, .8)
        inp = slim.max_pool2d(inp, (2, 2))
        return inp
        
    def _TransitionUp(self, inp, filters):
        inp = slim.conv2d_transpose(inp, filters, kernel_size=3, stride=2, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))
        return inp
    
    def _build_tiramisu(self, edgelength, CLASSES, BANDS, k, layers):
        self.X = tf.placeholder('float', shape=[None, edgelength, edgelength, BANDS])
        self.y = tf.placeholder('float', shape=[None, edgelength, edgelength, CLASSES])
        self.phase = tf.placeholder('bool')
        
        #w = tf.where(y > 0, 1, 0) # discard void class for later loss calculation
        
        concatenations = []
        
        cycle_start = slim.conv2d(self.X, 48, kernel_size=3, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32), scope='conv_init')        
                
        for l in layers[:-1]:
            blocked = self._DenseBlock(cycle_start, self.phase, l, k)
            concatenated = tf.concat([cycle_start, blocked], axis=3)
            concatenations.append(concatenated)
            cycle_start = self._TransitionDown(concatenated, self.phase)
        
        blocked = self._DenseBlock(cycle_start, self.phase, layers[-1], k)        
        
        for i, l in enumerate(layers[::-1][:-1]):
            transitioned_up = self._TransitionUp(blocked, l*k + layers[len(layers)-i-2]*k)
            concatenated = tf.concat([concatenations[4-i],transitioned_up], axis=3)
            blocked = self._DenseBlock(concatenated, self.phase, l, k)
            
        conv_final = slim.conv2d(blocked, CLASSES, 3, 1, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))
                
        return conv_final, self.y
        
    def _trainer(self, logits, labels, n_classes):
        softmaxed = tf.nn.softmax(logits)
        flat_logits = tf.reshape(softmaxed, [-1, n_classes])
        flat_labels = tf.reshape(self.y, [-1, n_classes])
        #flat_weights = tf.reshape(w, [-1, 1])
        #tf.losses.sparse_softmax_cross_entropy(labels=flat_labels, logits=flat_logits, weights=flat_weights)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels))        
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = tf.train.AdamOptimizer(1e-3, .995).minimize(self.loss)
        

