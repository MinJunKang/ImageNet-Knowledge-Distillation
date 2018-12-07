import Multitask
import Header
import threading
try:
    import urllib.request
    import urllib
    import tarfile
    from PIL import Image
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception as e:
    print("Error : ",e)
    exit(1)

class Dataset(object):
    def __init__(self,label_num,max_label_data_num):

        # For time checking
        start_time = Header.time.time()

        # Data Path Information
        self.zip_file = './imagenet_fall11_urls.tgz'
        self.url_text = './fall11_urls.txt'
        self.label_text = 'label.txt'
        self.raw_text = 'words.txt'
        self.data_dir = './data/'
        self.train_dir = './data/train/'
        self.valid_dir = './data/valid/'
        self.test_dir = './data/test/'
        self.label_dir = './label/'
        self.tmp_dir = './data/tmp/'
        self.sample_dir = './sample/'

        # Prepare dir
        Header.check_and_makedir(self.label_dir)
        Header.check_and_makedir(self.sample_dir)
        Header.check_and_makedir(self.data_dir)
        Header.check_and_makedir(self.train_dir)
        Header.check_and_makedir(self.valid_dir)
        Header.check_and_makedir(self.test_dir)
        Header.check_and_makedir(self.tmp_dir,True)

        # Setting
        self.label_num = label_num
        self.max_num = max_label_data_num
        self.data_num = 0 # initialize
        self.train_num = 0
        self.test_num = 0
        self.valid_num = 0

        # Multitask setting
        self.task = Multitask.Multitask(2000,50)

        # Prepare zip file, urls, label file
        list = self.Check_Download()
        self.Download(list)

        # Choose Label and collect the urls, Download the corresponding images
        if(self.Check_Data() == False):
            self.Data_Prepare()

        # Check the result and count the number of data
        self.Check_Result()

        print("\nElapsed time : %d sec" % (Header.time.time()-start_time))

    def Check_Download(self):
        Download_List = []
        # Check urls
        if(Header.os.path.isfile(self.url_text) == False):
            if(Header.os.path.isfile(self.zip_file) == False):
                Download_List.append(0)
            else:
                Download_List.append(1)

        # Check labels
        if(Header.os.path.isfile(self.label_dir + self.label_text) == False):
            if(Header.os.path.isfile(self.label_dir + self.raw_text) == False):
                Download_List.append(2)
            else:
                Download_List.append(3)
        else:
            num_label = len(open(self.label_dir + self.label_text,'rt').read().split('\n'))
            if(num_label != self.label_num + 1):
                if(Header.os.path.isfile(self.label_dir + self.raw_text) == False):
                    Download_List.append(2)
                else:
                    print("You choosed %d label but there exists file that contains %d label." % (self.label_num,num_label - 1))
                    x = input("Erase it or not? (Y/N) [If you choose Y, start from selecting labels] >> ")
                    x = x.lower()
                    if(x == "y"):
                        Download_List.append(3)
                    else:
                        self.label_num = num_label - 1

        return Download_List

    def Download(self,Download_List):
        for list in Download_List:
            if(list == 0):
                print("Download url zip file...")
                url = 'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz'
                urllib.request.urlretrieve(url, self.zip_file,Header.dlProgress)
                print("\n")
            if(list == 0 or list == 1):
                print("Unzip the url file...")
                try:
                    tar = tarfile.open(self.zip_file)
                    tar.extractall()
                    tar.close()
                except Exception as e:
                    print("Unzipping Error : ",e)
                    print("Try Again !! ")
                    Header.os.remove(self.zip_file)
                    exit(1)
            if(list == 2):
                print("Download label text file...")
                url = 'http://image-net.org/archive/words.txt'
                urllib.request.urlretrieve(url, self.label_dir+self.raw_text,Header.dlProgress)
                print("\n")
            if(list == 2 or list == 3):
                print("Extract %d label and save ..." % self.label_num)
                idx_list = []
                label_list = []
                names = ""
                with open(self.label_dir + self.raw_text,'rt') as f:
                    for line in f:
                        [idx,label_type] = Header.np.chararray.split(line,['\t'])[0]
                        idx_list.append(idx)
                        label_list.append(label_type)
                with open(self.url_text,'rt',encoding = "utf-8" ,errors='ignore') as f:
                    for line in f:
                        try:
                            [idx,url_jpg] = Header.np.chararray.split(line,['\t'])[0]
                            idx = Header.np.chararray.split(idx,['_'])[0][0]
                        except:
                            continue
                        if(idx != names):
                            self.task.begin_pool(Extract_Labels,input=[idx,idx_list])
                            names = idx
                    if(self.task.pool != None):
                        self.task.join_pool()
                idxs = self.task.Get_Pool_Result()
                new_idx_list = []
                new_label_list = []
                for id in idxs:
                    new_idx_list.append(idx_list[id])
                    new_label_list.append(label_list[id])
                idx_list,label_list = Header.Shuffle_Data(new_idx_list,new_label_list)
                with open(self.label_dir + self.label_text,'wt') as f:
                    for i in range(min(len(idx_list),self.label_num)):
                        f.write(idx_list[i] + "\t" + label_list[i])
                idx_list.clear()
                label_list.clear()
                new_idx_list.clear()
                new_label_list.clear()
                idxs.clear()

    def Check_Data(self):
        idxs = []
        print("Checking Dataset ... ")
        # Get label data
        with open(self.label_dir + self.label_text,'rt') as f:
            for line in f:
                idxs.append(Header.np.chararray.split(line,['\t'])[0][0])

        # Check train data set
        for name in Header.os.listdir(self.train_dir):
            self.task.begin_pool(Compare_Labels,input=[name,idxs])
        if(self.task.pool != None):
            self.task.join_pool()
        if(self.task.result_pool.qsize() != len(Header.os.listdir(self.train_dir))):
            return False
        idx_test = self.task.Get_Pool_Result()

        # Check valid data set
        for name in Header.os.listdir(self.test_dir):
            self.task.begin_pool(Compare_Labels,input=[name,idxs])
        if(self.task.pool != None):
            self.task.join_pool()
        if(self.task.result_pool.qsize() != len(Header.os.listdir(self.test_dir))):
            return False
        idx_test += self.task.Get_Pool_Result()

        # Check test data set
        for name in Header.os.listdir(self.valid_dir):
            self.task.begin_pool(Compare_Labels,input=[name,idxs])
        if(self.task.pool != None):
            self.task.join_pool()
        if(self.task.result_pool.qsize() != len(Header.os.listdir(self.valid_dir))):
            return False
        idx_test += self.task.Get_Pool_Result()
        idx_test = list(set(idx_test))

        print("# Train data : %d" % len(Header.os.listdir(self.train_dir)))
        print("# Test data : %d" % len(Header.os.listdir(self.test_dir)))
        print("# Valid data : %d" % len(Header.os.listdir(self.valid_dir)))
        print("Valid Label : %d / %d" % (len(idx_test),len(idxs)))
        
        if(len(idx_test) == len(idxs)):
            return True
        else:
            return False

    def Check_Result(self):
        idxs = []
        label_info = []
        print("Checking Dataset ... ")
        # Get label data
        with open(self.label_dir + self.label_text,'rt') as f:
            for line in f:
                idxs.append(Header.np.chararray.split(line,['\t'])[0][0])
                label = Header.np.chararray.split(line,['\t'])[0][1]
                label_info.append(Header.np.chararray.split(label,['\n'])[0][0])

        # Check train data set
        for name in Header.os.listdir(self.train_dir):
            self.task.begin_pool(Compare_Labels,input=[name,idxs])
        if(self.task.pool != None):
            self.task.join_pool()
        if(self.task.result_pool.qsize() != len(Header.os.listdir(self.train_dir))):
            return False
        idx_test = self.task.Get_Pool_Result()

        # Check valid data set
        for name in Header.os.listdir(self.test_dir):
            self.task.begin_pool(Compare_Labels,input=[name,idxs])
        if(self.task.pool != None):
            self.task.join_pool()
        if(self.task.result_pool.qsize() != len(Header.os.listdir(self.test_dir))):
            return False
        idx_test += self.task.Get_Pool_Result()

        # Check test data set
        for name in Header.os.listdir(self.valid_dir):
            self.task.begin_pool(Compare_Labels,input=[name,idxs])
        if(self.task.pool != None):
            self.task.join_pool()
        if(self.task.result_pool.qsize() != len(Header.os.listdir(self.valid_dir))):
            return False
        idx_test += self.task.Get_Pool_Result()
        idx_test = list(set(idx_test))

        if(len(idx_test) != len(idxs)):
            print("%d / %d labels are detected. Some labels don't have enough number." % (len(idx_test),len(idxs)))
            x = input("Try again or not? (Y/N) [If you choose Y, start from selecting labels] >> ")
            x = x.lower()
            if(x == "y"):
                Header.os.remove(self.label_dir + self.label_text)
                print("Try again")
            else:
                delete_count = 0
                new_idx = []
                for i in range(len(label_info)):
                    check = 0
                    for idx in idx_test:
                        if(idxs[i] == idx):
                            new_idx.append(idx)
                            check = 1
                            break
                    if(check == 0):
                        del label_info[i-delete_count]
                        delete_count += 1
                with open(self.label_dir + self.label_text,"wt") as f:
                    for i in range(len(label_info)):
                        f.write(new_idx[i] + "\t" + label_info[i] + "\n")
                print("%d labels are ready. Start again with %d label setting" % (len(label_info),len(label_info)))
        else:
            print("Data is perfect !")
            self.train_num = len(Header.os.listdir(self.train_dir))
            self.test_num = len(Header.os.listdir(self.test_dir))
            self.valid_num = len(Header.os.listdir(self.valid_dir))
            self.data_num = self.train_num + self.test_num + self.valid_num
            print("# Train data : %d" % self.train_num)
            print("# Test data : %d" % self.test_num)
            print("# Valid data : %d" % self.valid_num)
            print("# Overall data : %d" % self.data_num)


    def Data_Prepare(self):
        Header.check_and_makedir(self.train_dir,True)
        Header.check_and_makedir(self.valid_dir,True)
        Header.check_and_makedir(self.test_dir,True)

        # Prepare dummy data to remove
        dummy_sample = self.Check_Sample()

        # Get label data
        idxs = []
        with open(self.label_dir + self.label_text,'rt') as f:
            for line in f:
                idxs.append(Header.np.chararray.split(line,['\t'])[0][0])

        names = ""
        urls = []
        urls_idx = []
        url_test = []
        collect = False

        # Collect urls
        with open(self.url_text,'rt',encoding = "utf-8" ,errors='ignore') as f:
            for line in f:
                try:
                    [idx_,url_jpg] = Header.np.chararray.split(line,['\t'])[0]
                    idx = Header.np.chararray.split(idx_,['_'])[0][0]
                except:
                    continue
                if(idx != names):
                    collect = False
                    for x in idxs:
                        if(idx == x):
                            collect = True
                            break
                    names = idx
                if(collect):
                    urls.append(url_jpg)
                    urls_idx.append(idx_)
                    url_test.append(idx)

        # Download the images
        start_time = Header.time.time()
        now = Header.time.gmtime(start_time + 3600 * 9)
        print("Downloading Images ... Download %d urls" % len(urls))
        print("Start time : %d / %d / %d   %d : %d : %d" % (now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec))
        self.event = threading.Event()
        self.count_process = 0
        names = Header.np.chararray.split(urls_idx[0],['_'])[0][0]
        for i in range(len(urls)):
            if(Header.np.chararray.split(urls_idx[i],['_'])[0][0] != names):
                self.task.join_pool(timeout=1)
                result_list = self.task.Get_Pool_Result()
                for list in result_list:
                    if(list[0] == False):
                        if(Header.os.path.isfile(self.tmp_dir + list[1] + '.jpg')):
                            Header.os.remove(self.tmp_dir + list[1] + '.jpg')
                names = Header.np.chararray.split(urls_idx[i],['_'])[0][0]

            self.task.begin_pool(self.Image_Download,[urls_idx[i],urls[i],dummy_sample,len(urls),start_time])
        
        self.task.join_pool(timeout = 20)
        self.event.set()
        self.task.join_pool(timeout = 10,terminate_all=True)

        result_list = self.task.Get_Pool_Result()
        for list in result_list:
            if(list[0] == None):
                continue
            if(list[0] == False):
                if(Header.os.path.isfile(self.tmp_dir + list[1] + '.jpg')):
                    Header.os.remove(self.tmp_dir + list[1] + '.jpg')
        print("\n")

        urls_idx.clear()
        urls.clear()

        names = []
        for name in Header.os.listdir(self.tmp_dir):
            names.append(name)

        # Time elapsed
        elapsed = Header.time.gmtime(Header.time.time() - start_time)
        print("Elapsed time : %d mon %d day %d hour %d min %d sec" % (elapsed.tm_mon-1,elapsed.tm_mday-1,elapsed.tm_hour,elapsed.tm_min,elapsed.tm_sec))

        print("Refining Data ... Do not interrupt ... ")
        delete_count = 0
        for idx,name in enumerate(names):
            try:
                if(Header.os.path.isfile(self.tmp_dir + name)):
                    while(1):
                        if(Header.os.access(self.tmp_dir + name,Header.os.R_OK)):
                            break
                    im = None
                    im=Image.open(self.tmp_dir + name)
                    if(im != None):
                        im.close()
            except Exception as e:
                if(Header.os.path.isfile(self.tmp_dir + name)):
                    while(1):
                        if(Header.os.access(self.tmp_dir + name,Header.os.R_OK)):
                            break
                    Header.os.remove(self.tmp_dir + name)
                del names[idx-delete_count]
                delete_count += 1
        print("Splitting the Data ... Do not interrupt ... ")
        self.data_num = len(names)
        self.train_num = int(self.data_num * Header.np.divide(11,14))
        self.valid_num = int(self.data_num * Header.np.divide(1,14))
        self.test_num = self.data_num - (self.train_num + self.valid_num)

        Header.random.shuffle(names)
        name_train = names[0:self.train_num]
        name_valid = names[self.train_num:self.train_num + self.valid_num]
        name_test = names[self.train_num + self.valid_num:]
        names.clear()

        for name in name_train:
            new_name = Header.np.chararray.split(Header.np.chararray.split(name,['_'])[0][1],['.'])[0][0] + "_" + Header.np.chararray.split(name,['_'])[0][0] + "_" + str(self.Find_label_num(Header.np.chararray.split(name,['_'])[0][0],idxs)) + ".jpg"
            Header.shutil.move(self.tmp_dir + name,self.train_dir + new_name)
        for name in name_valid:
            new_name = Header.np.chararray.split(Header.np.chararray.split(name,['_'])[0][1],['.'])[0][0] + "_" + Header.np.chararray.split(name,['_'])[0][0] + "_" + str(self.Find_label_num(Header.np.chararray.split(name,['_'])[0][0],idxs)) + ".jpg"
            Header.shutil.move(self.tmp_dir + name,self.valid_dir + new_name)
        for name in name_test:
            new_name = Header.np.chararray.split(Header.np.chararray.split(name,['_'])[0][1],['.'])[0][0] + "_" + Header.np.chararray.split(name,['_'])[0][0] + "_" + str(self.Find_label_num(Header.np.chararray.split(name,['_'])[0][0],idxs)) + ".jpg"
            Header.shutil.move(self.tmp_dir + name,self.test_dir + new_name)

        idxs.clear()

        return

    def Check_Sample(self):
        pix_sample = []
        # Check if there is sample data
        if(len([name for name in Header.os.listdir(self.sample_dir) if Header.os.path.isfile(Header.os.path.join(self.sample_dir, name))]) != 3):
            print("Download Sample data...")
            Header.check_and_makedir(self.sample_dir)
            urllib.request.urlretrieve('http://farm3.static.flickr.com/2278/2300491905_5272f77e56.jpg', Header.os.path.join(self.sample_dir,'sample_1.jpg'))
            urllib.request.urlretrieve('http://www.epa.gov/bioiweb1/images/invertebrates/stonefly/stoneflies3.jpg', Header.os.path.join(self.sample_dir,'sample_2.jpg'))
            urllib.request.urlretrieve('http://bizhi.zhuoku.com/wall/20060714/WebshotsSeabed_1013.jpg', Header.os.path.join(self.sample_dir,'sample_3.jpg'))

        # Get sample info
        for i in range(3):
            im = Image.open(self.sample_dir+'sample_%s.jpg' % str(i+1))
            pix_sample.append(list(im.getdata()))
            im.close()
        return pix_sample

    # For Using Pool or Process
    def Image_Download(self,lock,input):
        idx = input[0]
        url = input[1]
        pix_sample = input[2]
        try:
            urllib.request.urlretrieve(url, Header.os.path.join(self.tmp_dir,idx + '.jpg'))
            self.count_process += 1
            with lock:
                if self.event.is_set():
                    Header.os.remove(self.tmp_dir+idx + '.jpg')
                    return
            try:
                if(Header.os.path.isfile(self.tmp_dir+idx + '.jpg')):
                    im = None
                    im=Image.open(self.tmp_dir+idx + '.jpg')
                    pix = list(im.getdata())
                    if im is not None:
                        im.close()
                    for i in range(len(pix_sample)):
                        if(pix == pix_sample[i]): # To get rid off the garbage
                            return False,idx
            except Exception as e: # To get rid off the wrong jpg:
                return False,idx
        except Exception as e:
            self.count_process += 1
            return False,idx
        with lock:
            Header.dlProgress_2(self.count_process,input[3],input[4])
        return True,idx

    # For Using Thread
    def Image_Download_2(self,lock,result,input):
        idx = input[0]
        url = input[1]
        pix_sample = input[2]
        try:
            urllib.request.urlretrieve(url, Header.os.path.join(self.tmp_dir,idx + '.jpg'))
            self.count_process += 1
            with lock:
                if self.event.is_set():
                    Header.os.remove(self.tmp_dir+idx + '.jpg')
                    return
            try:
                if(Header.os.path.isfile(self.tmp_dir+idx + '.jpg')):
                    im = None
                    im=Image.open(self.tmp_dir+idx + '.jpg')
                    pix = list(im.getdata())
                    if im is not None:
                        im.close()
                    for i in range(len(pix_sample)):
                        if(pix == pix_sample[i]): # To get rid off the garbage
                            if(result.full() == False):
                                result.put([False,idx])
                            return
            except Exception as e: # To get rid off the wrong jpg:
                if(result.full() == False):
                    result.put([False,idx])
                return
        except Exception as e:
            self.count_process += 1
            if(result.full() == False):
                result.put([False,idx])
            return
        if(result.full() == False):
            result.put([True,idx])
        with lock:
            Header.dlProgress_2(self.count_process,input[3],input[4])
        return

    # Default function
    def Find_label_num(self,idx,idx_list):
        for idx_num,list in enumerate(idx_list):
            if(idx == list):
                return idx_num

# For Using Acceleration
def Extract_Labels(lock,input):
    for i,x in enumerate(input[1]):
        if(input[0] == x):
            return i

def Compare_Labels(lock,input):
    idx = Header.np.chararray.split(input[0],['_'])[0][1]
    for x in input[1]:
        if(idx == x):
            return x

