'''
Authors:      Nguyen Tan Viet Tuyen, Oya Celiktutan
Email:       tan_viet_tuyen.nguyen@kcl.ac.uk
Affiliation: SAIR LAB, King's College London
Project:     LISI -- Learning to Imitate Nonverbal Communication Dynamics for Human-Robot Social Interaction

Python version: 3.6
'''

import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib
import inlib.Extmodels as md

class GAN_models(object):
    def __init__(self,
                batch_size,
                seq_length,
                audio_dims,
                motion_dims,
                embcontext_dims,
                embmotion_dims,
                rnn_size,
                num_layers,
                rnn_keep_list):

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.audio_dims = audio_dims
        self.motion_dims = motion_dims
        self.context_dims = embcontext_dims
        self.mot_ebd_dim = embmotion_dims
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.rnn_keep_list = rnn_keep_list
        self.mus_ebd_dim = 60
        self.n_z = 56 
        self.kernel_size = [1, 3]
        self.stride = [1, 2]
        self.act_type = 'lrelu'
        
    def _build_FC2mot_graph(self, mot_cell_FC2, mot_state_FC2, motion_inputs_FC2, scope):
        with tf.variable_scope(scope):
            fc1_weights = tf.get_variable('fc6', [self.motion_dims, 250], dtype=tf.float32)
            fc1_biases = tf.get_variable('bias6', [250], dtype=tf.float32)
            fc1_linear = tf.nn.xw_plus_b(motion_inputs_FC2, fc1_weights, fc1_biases, name='fc6_linear')
            fc1_relu = tf.nn.relu(fc1_linear, name='fc6_relu')

            fc2_weights = tf.get_variable('fc7', [250, 500], dtype=tf.float32)
            fc2_biases = tf.get_variable('bias7', [500], dtype=tf.float32)
            fc2_linear = tf.nn.xw_plus_b(fc1_relu, fc2_weights, fc2_biases, name='fc7_linear')

            (cell_output, mot_state_FC2) = mot_cell_FC2(fc2_linear, mot_state_FC2)
            output = tf.reshape(cell_output, [-1, self.rnn_size])

            fc3_weights = tf.get_variable('fc8', [self.rnn_size, 500], dtype=tf.float32)
            fc3_biases = tf.get_variable('bias8', [500], dtype=tf.float32)
            fc3_linear = tf.nn.xw_plus_b(output, fc3_weights, fc3_biases, name='fc8_linear')
            fc3_relu = tf.nn.relu(fc3_linear, name='fc8_relu')

            fc4_weights = tf.get_variable('fc9', [500, self.mot_ebd_dim], dtype=tf.float32)
            fc4_biases = tf.get_variable('bias9', [self.mot_ebd_dim], dtype=tf.float32)
            fc4_linear = tf.nn.xw_plus_b(fc3_relu, fc4_weights, fc4_biases, name='fc9_linear')
            fc4_relu = tf.nn.relu(fc4_linear, name='fc9_relu')

            return fc4_relu

    def _build_mot_rnn_graph(self, mus_inputs, init_step_mot, motion_inputs_FC2, z_noise, train_type, residual_connection=True):
        if train_type == 0:
            is_training = True
        else:
            is_training = False
        rnn_layer_idx = 0
        mus_cell = tf_contrib.rnn.MultiRNNCell(
            [self._get_lstm_cell(i, int(self.rnn_size/2), self.rnn_keep_list, is_training)
             for i in range(rnn_layer_idx, rnn_layer_idx+1)], state_is_tuple=True)
        self.mus_initial_state = mus_cell.zero_state(self.batch_size, dtype=tf.float32)
        mus_state = self.mus_initial_state

        rnn_layer_idx = 1
        mot_cell = tf_contrib.rnn.MultiRNNCell(
            [self._get_lstm_cell(i, self.rnn_size, self.rnn_keep_list, is_training)
             for i in range(rnn_layer_idx, rnn_layer_idx+self.num_layers)], state_is_tuple=True)

        self.mot_initial_state = mot_cell.zero_state(self.batch_size, dtype=tf.float32)
        mot_state = self.mot_initial_state
        last_step_mot = init_step_mot

        rnn_layer_idx = 1
        mot_cell_FC2 = tf_contrib.rnn.MultiRNNCell(
            [self._get_lstm_cell(i, self.rnn_size, self.rnn_keep_list, is_training)
             for i in range(rnn_layer_idx, rnn_layer_idx+self.num_layers)], state_is_tuple=True)

        self.mot_initial_state_FC2 = mot_cell.zero_state(self.batch_size, dtype=tf.float32)
        mot_state_FC2 = self.mot_initial_state_FC2

        motion_outputs = []
        audio_emb_outputs = []
        context_emb_outputs = []

        with tf.variable_scope("Generator"):
            for time_step in range(self.seq_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                    last_step_mot = mot_output

                mot_input_FC2 = motion_inputs_FC2[:, time_step, :]
                mot_fea_FC2 = self._build_FC2mot_graph(mot_cell_FC2, mot_state_FC2, mot_input_FC2, scope= "Generator")

                mus_input = mus_inputs[:, time_step, :, :]

                mus_fea, mus_state =  self._build_mus_graph(time_step, mus_cell, mus_state, mus_input, int(self.rnn_size/2), self.audio_dims, is_training)

                mot_input = last_step_mot
                all_input = tf.concat([mus_fea, mot_input, mot_fea_FC2, z_noise], 1, name='mus_mot_input')

                fc1_weights = tf.get_variable('fc1', [self.mus_ebd_dim + self.motion_dims +self.mot_ebd_dim+self.n_z, 500], dtype=tf.float32)
                fc1_biases = tf.get_variable('bias1', [500], dtype=tf.float32)
                fc1_linear = tf.nn.xw_plus_b(all_input, fc1_weights, fc1_biases, name='fc1_linear')
                fc1_relu = tf.nn.relu(fc1_linear, name='fc1_relu')

                fc2_weights = tf.get_variable('fc2', [500, 500], dtype=tf.float32)
                fc2_biases = tf.get_variable('bias2', [500], dtype=tf.float32)
                fc2_linear = tf.nn.xw_plus_b(fc1_relu, fc2_weights, fc2_biases, name='fc2_linear')

                (cell_output, mot_state) = mot_cell(fc2_linear, mot_state)
                output = tf.reshape(cell_output, [-1, self.rnn_size])

                fc3_weights = tf.get_variable('fc3', [self.rnn_size, 500], dtype=tf.float32)
                fc3_biases = tf.get_variable('bias3', [500], dtype=tf.float32)
                fc3_linear = tf.nn.xw_plus_b(output, fc3_weights, fc3_biases, name='fc3_linear')
                fc3_relu = tf.nn.relu(fc3_linear, name='fc3_relu')

                fc4_weights = tf.get_variable('fc4', [500, 100], dtype=tf.float32)
                fc4_biases = tf.get_variable('bias4', [100], dtype=tf.float32)
                fc4_linear = tf.nn.xw_plus_b(fc3_relu, fc4_weights, fc4_biases, name='fc4_linear')
                fc4_relu = tf.nn.relu(fc4_linear, name='fc4_relu')

                fc5_weights = tf.get_variable('fc5', [100, self.motion_dims], dtype=tf.float32)
                fc5_biases = tf.get_variable('bias5', [self.motion_dims], dtype=tf.float32)
                fc5_linear = tf.nn.xw_plus_b(fc4_relu, fc5_weights, fc5_biases, name='fc5_linear')
                if residual_connection:
                    mot_output = tf.add(mot_input, fc5_linear)
                else:
                    mot_output = fc5_linear

                motion_outputs.append(mot_output)
                audio_emb_outputs.append(mus_fea)
                context_emb_outputs.append(mot_fea_FC2)

        motion_outputs = tf.reshape(tf.concat(motion_outputs, axis=1), [self.batch_size, self.seq_length, self.motion_dims])
        audio_emb_outputs = tf.reshape(tf.concat(audio_emb_outputs, axis=1), [self.batch_size, self.seq_length, self.mus_ebd_dim])
        context_emb_outputs = tf.reshape(tf.concat(context_emb_outputs, axis=1), [self.batch_size, self.seq_length, self.mot_ebd_dim])

        return motion_outputs, audio_emb_outputs, context_emb_outputs

    @staticmethod
    def _mus_conv(inputs, kernel_shape, bias_shape, is_training, strides=[1, 2, 2, 1]):
        conv_weights = tf.get_variable('conv', kernel_shape,
                                        initializer=tf.truncated_normal_initializer())
        conv_biases = tf.get_variable('bias', bias_shape,
                                        initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(inputs,
                            conv_weights,
                            strides=strides,
                            padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        norm = tf.layers.batch_normalization(bias, axis=3,
                                                training=is_training)

        elu = tf.nn.elu(norm)
        return elu
    @staticmethod
    def _get_lstm_cell(rnn_layer_idx, hidden_size, rnn_keep_list, is_training):
        lstm_cell = tf_contrib.rnn.BasicLSTMCell(
            hidden_size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
        print('rnn_layer: ', rnn_layer_idx, rnn_keep_list[rnn_layer_idx])
        if is_training and rnn_keep_list[rnn_layer_idx] < 1:
            lstm_cell = tf_contrib.rnn.DropoutWrapper(lstm_cell,
                                                      output_keep_prob=rnn_keep_list[rnn_layer_idx])
        return lstm_cell

    def _build_mus_graph(self, time_step, mus_cell, mus_state, inputs, mus_hidden_size, mus_dim, is_training):
        print("mus_graph: ", time_step)
        with tf.variable_scope("FC1_mus_rnn"):
            fc_weights = tf.get_variable('fc', [mus_hidden_size, self.mus_ebd_dim],
                                         initializer=tf.truncated_normal_initializer())
            fc_biases = tf.get_variable('bias', [self.mus_ebd_dim],
                                        initializer=tf.zeros_initializer())
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            mus_input = self._build_mus_conv_graph(inputs, mus_dim, is_training)
            (cell_output, mus_state) = mus_cell(mus_input, mus_state)
            output = tf.reshape(cell_output, [-1, mus_hidden_size])
            fc_output = tf.nn.xw_plus_b(output, fc_weights, fc_biases)

        return fc_output, mus_state

    def _build_mus_conv_graph(self, inputs, mus_dim, is_training):
        print("mus_conv_graph")
        mus_input = tf.reshape(inputs, [-1, mus_dim, 5, 1]) 

        with tf.variable_scope('conv1'):
            elu1 = self._mus_conv(mus_input,
                                  kernel_shape=[mus_dim, 2, 1, 64],
                                  bias_shape=[64],
                                  is_training=is_training)

        with tf.variable_scope('conv2'):
            elu2 = self._mus_conv(elu1,
                                  kernel_shape=[1, 2, 64, 128],
                                  bias_shape=[128],
                                  is_training=is_training)

        with tf.variable_scope('conv3'):
            elu3 = self._mus_conv(elu2,
                                  kernel_shape=[1, 2, 128, 256],
                                  bias_shape=[256],
                                  is_training=is_training)

        with tf.variable_scope('conv4'):
            elu4 = self._mus_conv(elu3,
                                  kernel_shape=[1, 2, 256, 512],
                                  bias_shape=[512],
                                  is_training=is_training)
            mus_conv_output = tf.reshape(elu4, [-1, 512*3])

        return mus_conv_output
        
    def _build_Dis_rnn_graph(self, motion, audio_emb, context_emb, train_type, re_use=False):
        if train_type == 0:
            is_training = True
        else:
            is_training = False
        
        rnn_layer_idx = 1
        lstm_cell = tf_contrib.rnn.MultiRNNCell(
            [self._get_lstm_cell(i, self.rnn_size, self.rnn_keep_list, is_training)
            for i in range(rnn_layer_idx, rnn_layer_idx+self.num_layers)], state_is_tuple=True)
        with tf.variable_scope("Discriminator", reuse=re_use):

            ebd_all = tf.concat([motion, audio_emb, context_emb], 2, name='context_input')

            con_in = tf.unstack(ebd_all, self.seq_length, 1)
            output, _ = tf.contrib.rnn.static_rnn(lstm_cell, con_in, dtype=tf.float32)


            fc12_weights = tf.get_variable('fc12', [self.rnn_size, 500], dtype=tf.float32)
            fc12_biases = tf.get_variable('bias12', [500], dtype=tf.float32)
            fc12_linear = tf.nn.xw_plus_b(output[-1], fc12_weights, fc12_biases, name='fc12_linear')
            fc12_relu = tf.nn.relu(fc12_linear, name='fc12_relu')

            fc13_weights = tf.get_variable('fc13', [500, 100], dtype=tf.float32)
            fc13_biases = tf.get_variable('bias13', [100], dtype=tf.float32)
            fc13_linear = tf.nn.xw_plus_b(fc12_relu, fc13_weights, fc13_biases, name='fc13_linear')
            fc13_relu = tf.nn.relu(fc13_linear, name='fc13_relu')

            fc14_weights = tf.get_variable('fc14', [100, 50], dtype=tf.float32)
            fc14_biases = tf.get_variable('bias14', [50], dtype=tf.float32)
            fc14_linear = tf.nn.xw_plus_b(fc13_relu, fc14_weights, fc14_biases, name='fc14_linear')
            fc14_relu = tf.nn.relu(fc14_linear, name='fc14_relu')

            fc15_weights = tf.get_variable('fc15', [50, 1], dtype=tf.float32)
            fc15_biases = tf.get_variable('bias15', [1], dtype=tf.float32)
            fc15_linear = tf.nn.xw_plus_b(fc14_relu, fc15_weights, fc15_biases, name='fc15_linear')

            return fc15_linear
    
    def _build_Dis_time_cond_cnn_graph(self, motion, audio_emb, context_emb, is_shuffle=False, re_use=False):
        print('seg_time_cond_cnn_graph')
        mot_input = tf.reshape(motion, [self.batch_size, 1, self.seq_length, self.motion_dims])
        all_input = mot_input
        if is_shuffle:
            original_shape = all_input.get_shape().as_list()
            np.random.seed(1234567890)
            shuffle_list = list(np.random.permutation(original_shape[0]))
            all_inputs = []
            for i, idx in enumerate(shuffle_list):
                all_inputs.append(all_input[idx:idx+1, :, :, :])
            all_input = tf.concat(all_inputs, axis=0)

        conv_list_d = [[32,  self.kernel_size, self.stride, 'SAME', self.act_type],
                    [64,  self.kernel_size, self.stride, 'SAME', self.act_type, 'bn'],
                    [128, self.kernel_size, self.stride, 'SAME', self.act_type, 'bn'],
                    [256, self.kernel_size, self.stride, 'SAME', self.act_type, 'bn']]
        fc_list_d = [[1, '']]
        outputs = md.cnn(all_input, conv_list_d, fc_list_d, name='Discriminator', reuse=re_use)
        return outputs

    def G_rec_loss(self, real_action, fake_action):
        return tf.reduce_mean(tf.squared_difference(fake_action, real_action))

    def D_loss(self, d_real_action, d_fake_action):
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_action, labels=tf.ones_like(d_real_action)*0.98))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_action, labels=tf.zeros_like(d_fake_action)))
        d_loss_total = d_loss_real + d_loss_fake

        return d_loss_total
        
    def G_adv_loss(self, d_fake):
        adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
        return adv_loss

    @staticmethod
    def tf_diff_axis_1(a):
        return a[:,1:,:]-a[:,:-1,:]

    def vel_loss(self, real_action, fake_action):
        real_diff = tf.abs(self.tf_diff_axis_1(real_action))
        fake_diff = tf.abs(self.tf_diff_axis_1(fake_action))
        real_vel = tf.reduce_mean(real_diff)
        fake_vel = tf.reduce_mean(fake_diff)
        return tf.abs(real_vel-fake_vel)
    
    def motion_Encoder(self, motion_inputs_FC2, train_type):
        if train_type == 0:
            is_training = True
        else:
            is_training = False

        rnn_layer_idx = 1
        mot_cell = tf_contrib.rnn.MultiRNNCell(
            [self._get_lstm_cell(i, self.rnn_size, self.rnn_keep_list, is_training)
                for i in range(rnn_layer_idx, rnn_layer_idx+self.num_layers)], state_is_tuple=True)

        self.mot_initial_state = mot_cell.zero_state(self.batch_size, dtype=tf.float32)

        rnn_layer_idx = 1
        mot_cell_FC2 = tf_contrib.rnn.MultiRNNCell(
            [self._get_lstm_cell(i, self.rnn_size, self.rnn_keep_list, is_training)
                for i in range(rnn_layer_idx, rnn_layer_idx+self.num_layers)], state_is_tuple=True)

        self.mot_initial_state_FC2 = mot_cell.zero_state(self.batch_size, dtype=tf.float32)
        mot_state_FC2 = self.mot_initial_state_FC2
        context_emb_outputs = []

        with tf.variable_scope("Generator"):
            for time_step in range(self.seq_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                mot_input_FC2 = motion_inputs_FC2[:, time_step, :]
                # (batch_size, 128)
                mot_fea_FC2 = self._build_FC2mot_graph(mot_cell_FC2, mot_state_FC2, mot_input_FC2, scope= "Generator")
                context_emb_outputs.append(mot_fea_FC2)

        context_emb_outputs = tf.reshape(tf.concat(context_emb_outputs, axis=1), [self.batch_size, self.seq_length*self.mot_ebd_dim])

        return context_emb_outputs

