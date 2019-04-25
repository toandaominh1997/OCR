paths = list(open('/home/bigkizd/Downloads/hades/images_train.txt'))
labels = list(open('/home/bigkizd/Downloads/hades/lines_train.txt'))

f = open("hades-train.json", "w")
f.write("{")
idx = 0 
for path, label in zip(paths, labels):
    dest = path.split('/')[-1].replace('\n', '').replace('\r\n', '').replace('\\', '/')[6:]
    label = label.split(' ')[-1].replace('\n', '').replace('"', '\\"')
    f.write('\"{}\":\"{}\"'.format(dest, label))
    idx+=1
    if(idx!=len(paths)):
        f.write(',')
f.write("}")