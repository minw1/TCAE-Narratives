from randomtime_data import RT_Narrative_Data_Module

task_list = ["pieman","tunnel","lucy","prettymouth","milkywayoriginal","slumlordreach","notthefallintact","21styear","bronx","black","forgot"]
data_module = RT_Narrative_Data_Module(task_list)
dataloaders = data_module._setup_dataloaders()

counter = 0
for i in dataloaders[0]:
    print(i[1])
    counter += 1
    if counter == 1:
        break