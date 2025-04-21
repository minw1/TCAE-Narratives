from randomtime_data import RT_Narrative_Data_Module
import numpy as np
from utils import pos_tags

task_list = ["pieman","tunnel","lucy","prettymouth","milkywayoriginal","slumlordreach","notthefallintact","21styear","bronx","black","forgot"]
data_module = RT_Narrative_Data_Module(task_list, num_workers=0)



dataloaders = data_module._setup_dataloaders()

print("here")
for j in range(3):
    print(f"partition {j}")
    for idx, batch in enumerate(dataloaders[j]):
        print(f"Did batch {idx} in partition {j}")
        #print(batch[0].shape)
        print((batch[1].numpy() == pos_tags["PAD"]).argmax(axis=1))
        #print(batch[1][0].numpy())
        #print(np.mean(batch[0].numpy()))
        #print(np.mean(np.var(batch[0].numpy(), axis=2)))
print("all done")

'''
tunnels = data_module.create_tunnel()
for dset in tunnels:
    print(f"partition{dset.which_type} has {dset.segs_per_sub} segments, {dset.num_ex} examples")

tunnels[0].__getitem__(0)
'''