'''
Authors:      Nguyen Tan Viet Tuyen, Oya Celiktutan
Email:       tan_viet_tuyen.nguyen@kcl.ac.uk
Affiliation: SAIR LAB, King's College London
Project:     LISI -- Learning to Imitate Nonverbal Communication Dynamics for Human-Robot Social Interaction

Python version: 3.6
'''
'''
Personal notes
'''
import os
import numpy as np
import tensorflow as tf
import pickle5 as pickle
from sklearn.utils import shuffle
seq_length_train = 91 -1 
n_audio = 48 
n_motion = 30*3
n_embcontext = 256
n_embmotion = 128
n_z = 56
from models import GAN_models
from utils import get_30joints, APE_Acc_JERK

# Learning
tf.app.flags.DEFINE_float("learning_rate_G", .0002, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epoch", int(5e3), "epoch to train for.")
tf.app.flags.DEFINE_integer("start_decay", 500, "Start decay learning rate from this epoch.") 
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9, "Learning rate is multiplied by this much. 1 means no decay.") 
tf.app.flags.DEFINE_integer("learning_rate_step", 100, "Every this many epochs, do decay.") 
tf.app.flags.DEFINE_integer("warm_up", 0, "Initial training without adversarial loss") 
tf.app.flags.DEFINE_integer("save_every", 100, "How often to save the model.") 
tf.app.flags.DEFINE_integer("test_every", 100, "How often to save the model.") 
tf.app.flags.DEFINE_integer("start_saving", 100, "Start saving the model") 
tf.app.flags.DEFINE_float("rec_loss_weight", 10.0, "Reconstruction weights of Gface")
tf.app.flags.DEFINE_float("vel_loss_weight", 5.0, "Reconstruction weights of Gface")
tf.app.flags.DEFINE_float("adv_loss_weight", 1.0, "Reconstruction weights of Gface")
tf.app.flags.DEFINE_integer("rnn_size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_list("rnn_keep_list", [1.0, 1.0, 1.0], "rnn_keep_probability list, [1.0, 1.0, 1.0]")
tf.app.flags.DEFINE_string("restore_dir", " ", "Restore directory.") 
tf.app.flags.DEFINE_integer("restore_indx", 0, "Restore directory.") 


tf.app.flags.DEFINE_string("dataset", "../PreProcessData", "Training set.") 
tf.app.flags.DEFINE_string("model_dir", "./model_SessALL", "Training directory.")

FLAGS = tf.app.flags.FLAGS
summaries_dir = os.path.normpath(os.path.join(FLAGS.model_dir, "log"))

def create_session():
    return tf.Session(config=tf.ConfigProto(device_count={"CPU":3, "GPU":2}, 
                                            inter_op_parallelism_threads=0,
                                            allow_soft_placement=True,
                                            gpu_options={'allow_growth': True, 'visible_device_list': "0"},
                                            intra_op_parallelism_threads=0))

class GANtrainer():
    def __init__(self, 
                batch_size,
                learning_rate,
                summaries_dir,
                dtype=tf.float32):

        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * FLAGS.learning_rate_decay_factor)
        self.summaries_dir = summaries_dir

        self.init_step_mot = tf.placeholder(dtype, [self.batch_size, n_motion], name="init_pose") 
        self.motion_inputs = tf.placeholder(dtype, shape=[self.batch_size, seq_length_train, n_motion], name="motion_inputs") 
        self.audio_inputs  = tf.placeholder(dtype, shape=[self.batch_size, seq_length_train, n_audio*5, 1], name="audio_inputs") 
        self.motion_inputs_FC2 = tf.placeholder(dtype, shape=[self.batch_size, seq_length_train, n_motion], name="motion_inputs_FC2") 
        self.z = tf.placeholder(dtype, [self.batch_size, n_z], name="noise_vector") 

    def train(self):
        model = GAN_models(
            FLAGS.batch_size,
            seq_length_train,
            n_audio,
            n_motion,
            n_embcontext,
            n_embmotion,
            FLAGS.rnn_size, 
            FLAGS.num_layers,
            FLAGS.rnn_keep_list)

        motion_inputs = self.motion_inputs
        init_step_mot = self.init_step_mot
        audio_inputs = self.audio_inputs
        motion_inputs_FC2 = self.motion_inputs_FC2
        z_noise = self.z

        gen_motion, audio_emb, context_emb = model._build_mot_rnn_graph(audio_inputs, init_step_mot, motion_inputs_FC2, z_noise, train_type=0, residual_connection=False)
        gen_rec_loss = model.G_rec_loss(real_action=motion_inputs,
                                fake_action=gen_motion)
        gen_vel_loss = model.vel_loss(real_action=motion_inputs,
                                fake_action=gen_motion)
        
        d_real = model._build_Dis_time_cond_cnn_graph(motion_inputs, audio_emb, context_emb)
        d_fake = model._build_Dis_time_cond_cnn_graph(motion_inputs, audio_emb, context_emb, is_shuffle=True, re_use=True)

        dis_loss = model.D_loss(d_real, d_fake)
        gen_adv_loss = model.G_adv_loss(d_fake)
        gen_loss = gen_rec_loss*FLAGS.rec_loss_weight + gen_vel_loss*FLAGS.vel_loss_weight + gen_adv_loss*FLAGS.adv_loss_weight


        T_vars = tf.trainable_variables()
        G_vars = [var for var in T_vars if var.name.startswith('Generator')]
        D_vars = [var for var in T_vars if (var.name.startswith('Discriminator'))]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            g_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999).minimize(gen_loss, var_list=G_vars)
            d_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate/2, beta1=0.9, beta2=0.999).minimize(dis_loss, var_list=D_vars)

        gan_saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
        with open(os.path.join(FLAGS.model_dir, 'testing_eval.txt'), 'a') as f:
            info = zip(["dataset"], [FLAGS.dataset])
            for item in info:
                f.write(str(item) + " ")
            f.write("\n")


        with create_session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            if (FLAGS.restore_indx):
                checkpoint_EG = FLAGS.restore_dir + "/checkpoint-" + str(FLAGS.restore_indx) + "/checkpoint-" + str(FLAGS.restore_indx)
                gan_saver.restore(sess, checkpoint_EG)
                print("Restored model at: ", checkpoint_EG)

            t_ini_pose_FC1, t_motion_FC1, t_audio_FC1, t_motion_FC2 = data_generator(data_type="training")
            test_ini_pose_FC1, test_motion_FC1, test_audio_FC1, test_motion_FC2 = data_generator(data_type="testing")
            print("Saving model at ", FLAGS.model_dir)

            for epoch in range(FLAGS.restore_indx, FLAGS.epoch+1):
                G_losses = []
                D_losses = []
                for iter in range(np.shape(t_ini_pose_FC1)[0]//FLAGS.batch_size):
                    ini_pose_FC1_ = t_ini_pose_FC1[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                    motion_FC1_ = t_motion_FC1[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                    audio_FC1_ = t_audio_FC1[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                    motion_FC2_ = t_motion_FC2[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                    z_noise_ = np.random.normal(0, 1, (FLAGS.batch_size, n_z))

                    loss_D, _ = sess.run([dis_loss, d_trainer], feed_dict={self.init_step_mot: ini_pose_FC1_,
                                                                                self.motion_inputs: motion_FC1_,
                                                                                self.audio_inputs: audio_FC1_,
                                                                                self.motion_inputs_FC2: motion_FC2_,
                                                                                self.z: z_noise_})

                    loss_G, _ = sess.run([gen_loss, g_trainer], feed_dict={self.init_step_mot: ini_pose_FC1_,
                                                                                self.motion_inputs: motion_FC1_,
                                                                                self.audio_inputs: audio_FC1_,
                                                                                self.motion_inputs_FC2: motion_FC2_,
                                                                                self.z: z_noise_})
                    D_losses.append(loss_D)                    
                    G_losses.append(loss_G)

                print("epoch", epoch, "G_losses", np.mean(G_losses), "D_losses", np.mean(D_losses))
                with open(os.path.join(FLAGS.model_dir, 'training_loss.txt'), 'a') as f:
                    info = zip(["epoch", "G_losses", "D_losses"], [epoch, np.mean(G_losses), np.mean(D_losses)])
                    for item in info:
                        f.write(str(item) + " ")
                    f.write("\n")
            
                if epoch>FLAGS.start_decay and epoch % FLAGS.learning_rate_step == 0:
                    lr = sess.run(self.learning_rate_decay_op)
                    print("learning decay", lr)

                if epoch>= FLAGS.start_saving and epoch % FLAGS.save_every == 0:
                    gan_saver.save(sess, os.path.normpath(
                        os.path.join(FLAGS.model_dir, "checkpoint-"+str(epoch), 'checkpoint')), global_step=epoch)
                    print("Saving model at epoch", epoch)

                if epoch % FLAGS.test_every == 0:
                    for iter in range(np.shape(test_ini_pose_FC1)[0]//FLAGS.batch_size):
                        ini_pose_FC1_ = test_ini_pose_FC1[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        motion_FC1_ = test_motion_FC1[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        audio_FC1_ = test_audio_FC1[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        motion_FC2_ = test_motion_FC2[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
                        z_noise_ = np.random.normal(0, 1, (FLAGS.batch_size, n_z))

                        generated_FC1 = sess.run(gen_motion, feed_dict={init_step_mot: ini_pose_FC1_,
                                                                        audio_inputs: audio_FC1_,
                                                                        self.motion_inputs_FC2: motion_FC2_,
                                                                        self.z: z_noise_})
                        if iter == 0:
                            t_Gen_FC1 = generated_FC1
                            GT_FC1 = motion_FC1_
                        else:
                            t_Gen_FC1 = np.vstack((t_Gen_FC1,generated_FC1))
                            GT_FC1 = np.vstack((GT_FC1, motion_FC1_))
                    denorm_gen_mFC1 = denormalization_P(t_Gen_FC1)
                    denorm_mFC1 = denormalization_P(GT_FC1)

                    Ape, Acc, Jerk, Acc_GT, Jerk_GT = APE_Acc_JERK(t_GT=denorm_mFC1, t_Gen_FC1=denorm_gen_mFC1)                  
                    with open(os.path.join(FLAGS.model_dir, 'testing_eval.txt'), 'a') as f:
                        info = zip(["epoch", "APE", "Acc", "JERK", "Acc_GT", "Jerk_GT"], [epoch, Ape, Acc, Jerk, Acc_GT, Jerk_GT])
                        for item in info:
                            f.write(str(item) + " ")
                        f.write("\n")
                        
def denormalization_P(norm_data, P_max=400.0, P_min=-1200.0):
    eps = 1e-12
    denorm_data = (P_max-P_min+eps)*norm_data
    denorm_data = np.divide(denorm_data, 2) + P_min
    
    denorm_data = mm_to_cm(denorm_data)
    return denorm_data

def mm_to_cm(motion_data):
    motion_data = motion_data/10.0
    return motion_data

def data_generator(data_type):
    if data_type not in ['training', 'testing']:
        raise (ValueError, "'{0}' is not an appropriate data type.".format(data_type))
    SessALL = ['preprocess_Sess01', 'preprocess_Sess02', 'preprocess_Sess03', 'preprocess_Sess04', 'preprocess_Sess05']
    p_FC1_ALL = []
    p_FC2_ALL = []
    a_FC1_ALL = []
    for i, sess_i in enumerate(SessALL):
        dataset_sess_i =  os.path.join(FLAGS.dataset, sess_i, 'LISI.pickle')
        with open(dataset_sess_i, 'rb') as f:
            data = pickle.load(f)
            
            p_FC1_i = np.array(data['train_motion_FC1'], dtype="float32")  
            p_FC2_i = np.array(data['train_motion_FC2'], dtype="float32")  
            a_FC1_i = np.array(data['train_audio'], dtype="float32")
        p_FC1_ALL.append(p_FC1_i)
        p_FC2_ALL.append(p_FC2_i)
        a_FC1_ALL.append(a_FC1_i)
            
    p_FC1 = np.concatenate([p_FC1_ALL[0], p_FC1_ALL[1], p_FC1_ALL[2], p_FC1_ALL[3], p_FC1_ALL[4]])
    p_FC2 = np.concatenate([p_FC2_ALL[0], p_FC2_ALL[1], p_FC2_ALL[2], p_FC2_ALL[3], p_FC2_ALL[4]])
    a_FC1 = np.concatenate([a_FC1_ALL[0], a_FC1_ALL[1], a_FC1_ALL[2], a_FC1_ALL[3], a_FC1_ALL[4]])

    p_FC1 = get_30joints(p_FC1)
    p_FC2 = get_30joints(p_FC2)

    if data_type == 'training':
        num_train = int(np.shape(p_FC1)[0]*9/10)
        
        ini_pose_FC1 = p_FC1[0:num_train,0,:]
        motion_FC1 = p_FC1[0:num_train,1:seq_length_train+1,:]
        motion_FC2 = p_FC2[0:num_train,1:seq_length_train+1,:]
        audio_FC1  = a_FC1[0:num_train,1:seq_length_train+1,:]
        audio_FC1 = np.expand_dims(audio_FC1, axis=3)
        print("data_generator: ", data_type)
        print("ini_pose_FC1", np.shape(ini_pose_FC1), "motion_FC1: ", np.shape(motion_FC1), "audio_FC1: ", np.shape(audio_FC1))
        return ini_pose_FC1, motion_FC1, audio_FC1, motion_FC2

    if data_type == 'testing':        
        num_test = int(np.shape(p_FC1)[0]*1/10)

        ini_pose_FC1 = p_FC1[0:num_test,0,:]
        motion_FC1 = p_FC1[-num_test:,1:seq_length_train+1,:]
        motion_FC2 = p_FC2[-num_test:,1:seq_length_train+1,:]
        audio_FC1  = a_FC1[-num_test:,1:seq_length_train+1,:]
        audio_FC1 = np.expand_dims(audio_FC1, axis=3)
        print("data_generator: ", data_type)
        print("ini_pose_FC1", np.shape(ini_pose_FC1), "motion_FC1: ", np.shape(motion_FC1), "audio_FC1: ", np.shape(audio_FC1))
        return ini_pose_FC1, motion_FC1, audio_FC1, motion_FC2

if __name__ == "__main__":
    trainer = GANtrainer( FLAGS.batch_size,
                          FLAGS.learning_rate_G,
                          summaries_dir,
                          dtype=tf.float32)
    trainer.train()
