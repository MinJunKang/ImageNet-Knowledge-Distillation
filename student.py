

import Header
from PIL import Image
from PIL import ImageFile
from keras.applications.mobilenet import MobileNet

# No Dropout, Simple DNN Structure
class Student_model(object):
    def __init__(self, args, model_name):
        # Flag info
        self.flag = False

        self.model_name = model_name

        # Define Argument
        self.Define_Arg(args)

        # Training Parameters
        self.num_parameter = 0

        # To determine if it is trainable
        self.trainable = False

        self.last_epoch = 0
        self.input_shape =[224,224,3]
        self.output_shape = [50]


    def Define_Arg(self,args):
        # Temperature
        self.temperature = args.temperature

        # Learning parameter
        self.min_learning_rate = args.min_learning_rate
        self.learning_rate_increment = args.learning_rate_increment
        self.max_learning_rate = args.max_learning_rate
        self.min_batch = args.min_batch
        self.batch_increment = args.batch_increment
        self.max_batch = args.max_batch

        # Structure parameter
        self.min_layer = args.min_layer
        self.layer_increment = args.layer_increment
        self.max_layer = args.max_layer

        self.min_node = args.min_node
        self.node_increment = args.node_increment
        self.max_node = args.max_node
        
        # Set epoch and stop point criterion
        self.max_epoch = args.max_epoch
        self.max_overfit = args.max_overfit

        # model type
        self.model_type = args.model_type

        # Saving place
        self.dir = args.dir
        self.checkpoint_dir = args.checkpoint
        self.final_dir = args.final
        self.log_dir =args.log
        self.model_dir = args.model
        self.tmp_dir = args.tmp

        self.checkpoint_file = Header.os.path.join(self.checkpoint_dir,"checkpoint_weight.h5")
        self.final_file = Header.os.path.join(self.final_dir,"trained_weight.h5")
        self.log_file_1 = Header.os.path.join(self.log_dir,"Result_model.txt")
        self.model_file = Header.os.path.join(self.model_dir,"model.json")
        self.tmp_file_1 = Header.os.path.join(self.tmp_dir,"model.json")
        self.tmp_file_2 = Header.os.path.join(self.tmp_dir,"trained_weight.h5")
        self.tmp_file_3 = Header.os.path.join(self.tmp_dir,"info.txt")

        self.num_class = 50
    # Initialize the weight and bias
    def get_stddev(self,in_dim, out_dim):
        return 1.3 / Header.math.sqrt(float(in_dim) + float(out_dim))

    # Loss function
    def loss(self,y_true,y_pred):
        if(self.model_type == "classifier"):
            pred = Header.K.softmax(y_pred)

            total_loss = Header.tf.reduce_mean(Header.K.categorical_crossentropy(y_true,pred))

            loss_op_soft = Header.tf.cond(Header.tf.constant(self.flag),
                                        true_fn=lambda: Header.tf.reduce_mean(Header.K.categorical_crossentropy(
                                           self.soft_true, Header.K.softmax(y_pred / self.temperature))),
                                        false_fn=lambda: 0.0)
        else:
            pred = Header.K.tanh(y_pred)

            total_loss = Header.tf.reduce_mean(Header.tf.square(pred - y_true))

            loss_op_soft = Header.tf.cond(Header.tf.constant(self.flag),
                                        true_fn=lambda: Header.tf.reduce_mean(Header.tf.square(Header.K.tanh(y_pred/ self.temperature) - self.soft_true)),
                                        false_fn=lambda: 0.0)

        total_loss += Header.tf.square(self.temperature) * loss_op_soft

        return total_loss

    def loss_train_student(self,y_true,y_pred):  ##loss 筌띿쉶?쀯쭪  ?類ㅼ뵥 
        if(self.model_type == "classifier"):
            pred = Header.K.softmax(y_pred)

            total_loss = Header.tf.reduce_mean(Header.K.categorical_crossentropy(y_true,pred))
        else:
            pred = Header.K.tanh(y_pred)

            total_loss = Header.tf.reduce_mean(Header.tf.square(pred - y_true))
        return total_loss
    
    def acc_model(self,y_true,y_pred):
        if(self.model_type == "classifier"):
            prediction = Header.K.softmax(y_pred)
            accuracy = Header.tf.reduce_mean(Header.tf.cast(Header.tf.equal(Header.tf.argmax(prediction,1),Header.tf.argmax(y_true,1)),"float32"),name = "Accuracy_classifier")
        else:
            accuracy = Header.tf.reduce_mean(Header.tf.square(y_pred - y_true,name = "squared_accuracy"),name = "Accuracy_regression")
        return accuracy

    def Load_Text(self,dst_file):
        contents = []
        with open(dst_file,'rb') as myloaddata:
            contents = Header.pickle.load(myloaddata)
        return contents

    def Build_DNN_eval(self,hidden_units=[256,256]):

#        soft_y = Header.l.Input(shape=self.output_shape,name="%s_soft_y" % self.model_name)
#        self.soft_true = soft_y

        model = MobileNet(input_shape=(224,224,3), alpha=1.0, weights = None, depth_multiplier=1,include_top = False)
        out = Header.l.AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding="same")(model.output)
        out2 = Header.l.Flatten()(out)
        model = Header.m.Model(model.input,Header.l.Dense(50)(out2))
        

        return model

    def Build_DNN(self,hidden_units=[256,256]):

        soft_y = Header.l.Input(shape=self.output_shape,name="%s_soft_y" % self.model_name)
        self.soft_true = soft_y

        model = MobileNet(input_shape=(224,224,3), alpha=1.0, weights = None, depth_multiplier=1,include_top = False)
        out = Header.l.AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding="same")(model.output)
        out2 = Header.l.Flatten()(out)
        model = Header.m.Model([model.input,soft_y],Header.l.Dense(50)(out2))
        

        return model

    def Train(self,args,train_dir,valid_dir,train_num,valid_num,teacher_flag = False): ## teahcer_flag ??已?癰?野?
     
        # Define Argument
        self.Define_Arg(args)
        soft_file = args.softy_file

        self.in_dim = 224*224*3
        self.out_dim = 50

        f = open(self.log_file_1,'w')
        f.write("Temperature : %f\n" % self.temperature)
        f.close()
        
        # Change the structure
        for layer_num in range(self.min_layer,self.max_layer+1):
            hidden_node = []
            for layer in range(layer_num):
                hidden_node.append(self.min_node)
            for layer in range(layer_num):
                if(layer != 0):
                    hidden_node[layer] *= self.node_increment
                while(hidden_node[layer] <= self.max_node):
                    f = open(self.log_file_1,'a')
                    result=self.Optimize(hidden_node,teacher_flag,train_dir,valid_dir,soft_file,train_num,valid_num)
                    f.write("-------------------------------\n")
                    f.write("Model : [")
                    for i in range(len(hidden_node)):
                        f.write("%d  "%hidden_node[i])
                    f.write("]\nTraining_info : [")
                    f.write("learning_rate : %f     "%result[0])
                    f.write("batch_size : %d    " % result[1])
                    f.write("Last epoch : %d]\n" % self.last_epoch)
                    f.write("Train accuracy : %f        Validation accuracy : %f\n" % (result[3],result[2]))
                    f.write("Number of parameter : %d\n" % self.num_parameter)
                    f.write("-------------------------------\n")
                    f.close()
                    hidden_node[layer] *= self.node_increment
                    if(result[2] == 1.00):
                        print("Found the best model!! Finish the training")
                        return
                hidden_node[layer] = int(hidden_node[layer] / self.node_increment)


    def Optimize(self,hidden_node,teacher_flag,train_dir,valid_dir,soft_file,train_num,valid_num):
        rate = self.min_learning_rate
        result = 0
        self.batch = self.min_batch
        data = []
        data.append(rate)
        data.append(self.min_batch)
        data.append(result) # Test accuracy
        data.append(result) # Train accuracy

        [train_result,tmp_result] = self.Run(hidden_node,rate,self.batch,result,teacher_flag,train_dir,valid_dir,soft_file,train_num,valid_num)
        if(result < tmp_result):
            data[0] = rate
            data[1] = self.min_batch
            data[2] = tmp_result
            data[3] = train_result
            result = tmp_result
            if(result == 1.00):
                return data

        return data

    # Still Programming
    def Run(self,hidden_node,learning_rate,batch_size,result,teacher_flag,train_dir,valid_dir,soft_file,train_num,valid_num):

        # Build Model
        model = self.Build_DNN(hidden_node)

        test_acc_save = 0
        last_epoch = 0

        # compile model
        self.adam = Header.op.Adam(lr=learning_rate,clipvalue=1.5)

        train_gen = self.generate_arrays_from_file(train_dir,soft_file,batch_size)
        valid_gen = self.generate_arrays_from_file(valid_dir,soft_file,batch_size)

        train_data_count = train_num
        valid_data_count = valid_num

        train_per_epoch = round(train_data_count / batch_size + 0.5)
        valid_per_epoch = round(valid_data_count / batch_size + 0.5)
        
        if teacher_flag: ## teacher_flag ??뽰뒠
            model.compile(self.adam,loss=self.loss,metrics=[self.acc_model])
            self.flag = True 
        else:
            model.compile(self.adam,loss=self.loss_train_student,metrics=[self.acc_model])
            self.flag = False

        # Model information
        model.summary()
        self.num_parameter = int(Header.np.sum([Header.K.count_params(p) for p in set(model.trainable_weights)]))

        checkpointer = Header.call.ModelCheckpoint(filepath=self.checkpoint_file,monitor = "val_acc_model", verbose=1, save_best_only=True, save_weights_only=True)
        earlyStopping=Header.call.EarlyStopping(monitor='val_acc_model', patience=self.max_overfit, verbose=0, mode='auto')

        history = model.fit_generator(train_gen,steps_per_epoch = train_per_epoch,
                                      epochs = self.max_epoch, verbose = 2,callbacks=[checkpointer,earlyStopping],
                                      validation_data = valid_gen, validation_steps = valid_per_epoch,
                                      max_queue_size =100 , shuffle = False, initial_epoch = 0)
        # Get the value
        test_acc_save = Header.np.max(history.history['val_acc_model'])
        last_epoch = Header.np.argmax(history.history['val_acc_model'])+1
        test_loss_save = history.history['val_loss'][last_epoch-1]
        train_acc_save = history.history['acc_model'][last_epoch-1]
        train_loss_save = history.history['loss'][last_epoch-1]
        
        print("\n\nTrained Model Structure : ")
        print("Structure : ",hidden_node, "     Number of parameter : ",self.num_parameter)
        print("Train_parameter = learning_rate : ",learning_rate,"    batch_size : ",batch_size)
        print("Last Epoch : %d" % last_epoch)
        print("Validation Accuracy : ",test_acc_save,"      Validation loss : ",test_loss_save)
        print("Trainset Accuracy : ",train_acc_save,"       Trainset loss : ",train_loss_save)

        if(result < test_acc_save):
            self.last_epoch = last_epoch
            model.save(self.model_file)
            Header.shutil.copy(self.checkpoint_file,self.final_file)

        # return the result
        return [train_acc_save,test_acc_save]
   
    def Evaluate(self,args,test_dir,soft_file,test_num):
        if(Header.os.path.isfile(self.model_file) == False):
            print("Train the teacher model first!!")
            exit(1)

        self.Define_Arg(args)

        loaded_model_json = self.Build_DNN_eval() #Header.m.load_model(self.model_file,custom_objects = {'loss':self.loss,'acc_model':self.acc_model})
        batch_size = self.min_batch

        soft_y = Header.l.Input(shape=self.output_shape,name="%s_soft_y" % self.model_name)
        self.soft_true = soft_y


         # Load Model
        loaded_model = Header.m.model_from_json(loaded_model_json.to_json())
        loaded_model.load_weights(self.final_file)

        adam = Header.op.Adam(lr=0.00001,clipvalue=1.5)
        loaded_model.compile(loss = self.loss, optimizer = adam,metrics = [self.acc_model])

        test_gen =  self.generate_arrays_from_file(test_dir,soft_file,batch_size,ev = True)
        test_data_count = test_num
        test_per_epoch = round(test_data_count / batch_size + 0.5)
        return loaded_model.evaluate_generator(test_gen,test_per_epoch)
    
    def Num_Parameter(self):
        if(self.num_parameter != 0):
            return self.num_parameter

    def Fit_SingleModel(self,train_dir,valid_dir,test_dir,soft_file,train_num,valid_num,test_num,hidden_node = [256,256],learn_rate = 0.001,batch = 128):
        

        # Build Model
        model = self.Build_DNN(hidden_node)

        model.compile(self.adam,loss=self.loss,metrics=[self.acc_model])

        batch_size = self.min_batch
       
        train_gen = self.generate_arrays_from_file(train_dir,soft_file,batch_size)
        valid_gen = self.generate_arrays_from_file(valid_dir,soft_file,batch_size)
        test_gen =  self.generate_arrays_from_file(test_dir,soft_file,batch_size)

        train_data_count = train_num
        valid_data_count = valid_num
        test_data_count = test_num

        train_per_epoch = round(train_data_count / batch_size + 0.5)
        valid_per_epoch = round(valid_data_count / batch_size + 0.5)
        test_per_epoch = round(test_data_count / batch_size + 0.5)

        # Model information
        model.summary()
        parameter_num = int(Header.np.sum([Header.K.count_params(p) for p in set(model.trainable_weights)]))
        self.num_parameter = parameter_num

        checkpointer = Header.call.ModelCheckpoint(filepath=self.tmp_file_2,monitor = "val_acc_model", verbose=1, save_best_only=True, save_weights_only=True)
        earlyStopping=Header.call.EarlyStopping(monitor='val_acc_model', patience=self.max_overfit, verbose=0, mode='auto')

        history = model.fit_generator(train_gen,steps_per_epoch = train_per_epoch,
                                      epochs = self.max_epoch, verbose = 2,callbacks=[checkpointer,earlyStopping],
                                      validation_data = valid_gen, validation_steps = valid_per_epoch,
                                      max_queue_size =100 , shuffle = False, initial_epoch = 0)

        # Get the value
        valid_acc_save = Header.np.max(history.history['val_acc_model'])
        last_epoch = Header.np.argmax(history.history['val_acc_model'])+1
        valid_loss_save = history.history['val_loss'][last_epoch-1]
        train_acc_save = history.history['acc_model'][last_epoch-1]
        train_loss_save = history.history['loss'][last_epoch-1]

        # Save model
        model.save(self.tmp_file_1)

        # Inference
        result = model.evaluate_generator(test_gen,test_per_epoch,verbose=3)
        test_acc_save = result[1]

        # Result view
        model.summary()

        # Save the result to txt file
        with open(self.tmp_file_3,'wt') as f:
            f.write("Temperature : %f\n" % self.temperature)
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write("----------------------------------------\n")
            f.write("Training Parameter Information : \n")
            f.write("learning_rate : %f     batch_size : %d\n" % (learn_rate,self.min_batch))
            f.write("Last Epoch : %d\n" % last_epoch)
            f.write("----------------------------------------\n")
            f.write("Train Accuracy : %f    Train loss : %f\n" % (train_acc_save,train_loss_save))
            f.write("Valid Accuracy : %f    Test Accuracy : %f\n" % (valid_acc_save,test_acc_save))
            f.write("----------------------------------------\n")
        return [test_acc_save,parameter_num]

    def Temp_file3_add(self,data):
        with open(self.tmp_file_3,'at') as f:
            f.write("Complexity_gain(Teacher params / Student params) : %f\n" % data[0])
            f.write("Accuracy_gain(Teacher acc - Student acc) : %f\n" % data[1])
            f.write("Overall_gain(Complexity_gain / Accuracy_gain) : %f\n" % data[2])

    def View_Temp_log(self):
        if(Header.os.path.isfile(self.tmp_file_3) == False):
            print("You must first train the temporary student model to get spec")
            return False
        else:
            with open(self.tmp_file_3) as f:
                for line in f:
                    print(line)
            return True

    def View_Train_log(self):
        if(Header.os.path.isfile(self.log_file_1) == False):
            print("You must first train the student model to get train log")
            return False
        else:
            with open(self.log_file_1) as f:
                for line in f:
                    print(line)
            return True
    def generate_arrays_from_file(self,feature_dir,soft_file,batch_size,ev = False):

        length = len(Header.os.listdir(feature_dir))
        
        if soft_file is not None:
            with open(soft_file,'rb') as f1:
                soft_target = Header.pickle.load(f1)
        while 1:
            x, y, z= [],[],[]
            i = 0
            for index,name in enumerate(Header.os.listdir(feature_dir)):
                im = None
                im = Image.open(feature_dir+'/'+name)
                im = im.convert('RGB')
                im = im.resize((224,224))
                pix = list(im.getdata())
                pix = Header.np.asarray(pix)
                pix = Header.np.reshape(pix,(224,224,3))
                if(im != None):
                    im.close()
                label = int(Header.np.chararray.split(Header.np.chararray.split(name,['_'])[0][2],['.'])[0][0])
                y.append(label)
                x.append(pix)
                if soft_file is not None:
                    z.append(soft_target[index])
                else:
                    z.append(label)
                i += 1
                if i == batch_size:
                    y = Header.np_utils.to_categorical(y,self.num_class)
                    if soft_file is None:
                        z = Header.np_utils.to_categorical(z,self.num_class)
                    if ev is True:
                        yield (Header.np.array(x),Header.np.array(y))
                    else:
                        yield ([Header.np.array(x),Header.np.array(z)],Header.np.array(y))
                    i = 0
                    x, y, z = [], [], []
                elif(index == length - 1):
                    y = Header.np_utils.to_categorical(y,self.num_class)
                    if soft_file is None:
                        z = Header.np_utils.to_categorical(z,self.num_class)
                    if ev is True:
                        yield (Header.np.array(x),Header.np.array(y))
                    else:
                        yield ([Header.np.array(x),Header.np.array(z)],Header.np.array(y))
                    i = 0
                    x, y, z = [], [], []

