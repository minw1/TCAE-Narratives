from randomtime_data import RT_Narrative_Data_Module

task_list = ["pieman","tunnel","lucy","prettymouth","milkywayoriginal","slumlordreach","notthefallintact","21styear","bronx","black","forgot"]
data_module = RT_Narrative_Data_Module(task_list, num_workers=0)



dataloaders = data_module._setup_dataloaders()

print("here")
for j in range(3):
    print(f"partition {j}")
    for idx, batch in enumerate(dataloaders[j]):
        print(f"Did batch {idx} in partition {j}")
print("all done")

'''
tunnels = data_module.create_tunnel()
for dset in tunnels:
    print(f"partition{dset.which_type} has {dset.segs_per_sub} segments, {dset.num_ex} examples")

tunnels[0].__getitem__(0)
'''