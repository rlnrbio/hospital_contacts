# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 12:44:50 2022

@author: rapha
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

nocontact10s = True

# load and save actor and calls data 
path = "./detailed_list_of_contacts_Hospital.dat"
with open(path) as f:
    lines = f.readlines()
df = pd.DataFrame(columns = ["time", "sender", "receiver", "role1", "role2"])
for line in lines:
    l = line.replace("\n", "").split("\t")
    temp = pd.DataFrame([l], columns = ["time", "sender", "receiver", "role1", "role2"])
    df = df.append(temp)
    
calls = df[["time", "sender", "receiver"]]
actors1 = df[["sender", "role1"]]
a1 = actors1.drop_duplicates().rename(columns = {"sender":"actor", "role1":"role"})
actors2 = df[["receiver", "role2"]]
a2 = actors2.drop_duplicates().rename(columns = {"receiver":"actor", "role2":"role"})

actors = a2.append(a1).drop_duplicates().sort_values(by = "actor")
actors["recode"] = range(0,actors.shape[0])
actors.to_csv("./actors.csv", index = False)
calls.to_csv("./calls.csv", index = False)

# split data into day contact matrices
times = list(calls["time"])
starttime = int(times[0])
times = [starttime]
fullday = [1,2,3,4]
for day in fullday:
    tempdf = np.zeros(shape = (actors.shape[0],actors.shape[0]))
    endtime = starttime + (24*3600)

    index = np.array(calls["time"]).astype(int)
    index = np.logical_and(index > starttime, index <= endtime)
    
    daydf = calls[index]
    unique_conts = daydf[["sender", "receiver"]].drop_duplicates()
    recode_dict = dict(zip(actors["actor"], actors["recode"]))
    for index, row in unique_conts.iterrows():
        tempdf[recode_dict[row["sender"]], recode_dict[row["receiver"]]] = 1
        tempdf[recode_dict[row["receiver"]], recode_dict[row["sender"]]] = 1
    print(np.sum(tempdf))
    # mark individuals that did not have any contact
    if nocontact10s:
        index = np.sum(tempdf, axis = 0)==0
        tempdf[index] = np.full(tempdf[index].shape, 10)
    pd.DataFrame(tempdf.astype(int), columns = actors["actor"], index = actors["actor"]).to_csv("./network_day{}.csv".format(day))
    starttime = endtime
    times.append(endtime)

# create figure for contact distribution across days 
fig, ax = plt.subplots(figsize=(16,8))
plt.hist(calls["time"], bins = 97) # Data comprises 97 hours 
for t in range(0,5):
    plt.axvline(x = t*24*180*20, color = "r")
fig.savefig("./distribution.jpg")

roles = ["NUR", "MED", "PAT", "ADM"]
# calculate average nuber of contacts for each group:
for i, r in enumerate(roles): 
    for j in range(1, 5):
        day = pd.read_csv("./network_day{}.csv".format(j), index_col = "actor")
        #day.index = day["actor"]
        #day = day.drop("actor", axis = 1)
        acts = actors["role"] == r
        day = np.array(day)[acts]
        day = day[day[:,0]!=10]
        summed_contacts = np.sum(day, axis = 1)
        avg_daily_contacts = np.mean(summed_contacts)
        print("Avg_daily_contacts: Day: {}, {}: {}".format(j,r,avg_daily_contacts))
        


calls = pd.read_csv("./calls.csv")
actors = pd.read_csv("./actors.csv")


def time_agg(array, axis = 1, agg_elems = 15, agg_type = "or"):
    """
    Function to time aggregate a binary numpy array along an axis

    Parameters
    ----------
    array : nd_array
        The numpy array to be aggregated.
    axis : int, optional
        The axis along which the array should be updated. The default is 1.
    agg_elems : int, optional
        The number of elements that should be aggregated. The default is 15.
    agg_type : str, optional
        The type of aggregation that should be performed. The default is "or" 

    Returns
    -------
    The aggregated array.

    """
    new_shape = list(array.shape)
    new_shape[axis] = math.ceil(array.shape[axis]/agg_elems)
    new_array = np.empty(shape = tuple(new_shape))
    for ind, start in enumerate(range(0,array.shape[axis], agg_elems)):
        end = start + agg_elems
        temp_arr = array[:,start:end]
        new_array[:, ind] = temp_arr.any(axis)
        #print(start, end, temp_arr.any())
    return new_array

actorstring = [str(elem[0]) + ": " + elem[1] for elem in (zip(actors["actor"], actors["role"]))]

timepoints = np.array(range(0, np.array(calls["time"])[-1] + 20, 20))
recode_tp_dict = dict(zip(timepoints, range(0, timepoints.shape[0])))
shift_array = np.zeros(shape = (actors.shape[0], timepoints.shape[0]))

shift_pd = pd.DataFrame(shift_array, columns = timepoints)
for i, row in actors.iterrows():
    act = row["actor"]
    print(act)
    index = np.logical_or(calls["sender"] == act, calls["receiver"] == act) 
    present = calls[index]
    for j, p in present.iterrows():
        shift_array[i, recode_tp_dict[p["time"]]] = 1
fig, ax = plt.subplots(figsize=(26,15))        
sparse_hm = sns.heatmap(shift_array, yticklabels = actorstring)
for t in range(0,5):
    plt.axvline(x = t*24*180*20, color = "r")
fig.savefig("./Shifts_original.jpg")


def counter(index_vector):
    lenghts, positions = [],[]
    current_start = index_vector[0]
    current = index_vector[0]
    count = 1
    for i in index_vector[1:]:
        if i-current == 1:
            count += 1
            current = i
        else:
            lenghts.append(count)
            positions.append(current_start)
            current_start = i
            current = i
            count = 1
    lenghts.append(count)
    positions.append(current_start)

    return np.array(lenghts), np.array(positions)

def fill_space(vector, max_fill_length = 3):
    indices = np.where(vector == 0)[0]
    lens, pos = counter(indices)
    len_ind = lens <= max_fill_length
    for curr_len, curr_pos in zip(lens[len_ind], pos[len_ind]):
        if curr_pos == 0:
            # do not fill leading missing zeros
            continue
        if curr_pos + curr_len == len(vector):
            # do not fill lagging missing zeros
            continue
        vector[curr_pos:curr_pos+curr_len] = 1
    return vector
    
        
            
        
        
# create shift data:
#     
def shift_agg(array, actors):
    """
    Function to create start and end data for shifts in a format for DYNAM PACKAGE
    Using assumption of hourly shifts for medical staff and administration staff
    Assuming continuous occupation of patients on station
    Parameters
    ----------
    array : nd_array
        The numpy array to be aggregated.
    Returns
    -------
    The aggregated array.

    """
    hourly_array = time_agg(array, axis = 1, agg_elems = 45*4)
    hourly_array_full = np.zeros(shape = hourly_array.shape)
    shift_collection = pd.DataFrame(columns = ["time", "actor", "update"])
    hour_factor = 3600
    
    for i in range(actors.shape[0]):
        act = actors["actor"][i]
        role = actors["role"][i]
        contacts = hourly_array[i,:]
        if role == "PAT":
            # assume patients are present from first to last contact on the station
            contact_indices = np.where(contacts == 1)[0]
            first_contact = contact_indices[0]
            last_contact = contact_indices[-1]
            hourly_array_full[i,first_contact:last_contact +1] = 1
            
            
            # create attendence start and end data for dynam
            filled_contacts = hourly_array_full[i,]
            indices = np.where(filled_contacts == 1)[0]
            lens, pos = counter(indices)
            for curr_len, curr_pos in zip(lens, pos):
                # Add shiftstart and shiftend to df
                shift_collection = shift_collection.append(pd.DataFrame([[curr_pos*hour_factor, act, True]], 
                                                           columns = ["time", "actor", "update"]), ignore_index = True)
                shift_collection = shift_collection.append(pd.DataFrame([[(curr_pos+curr_len)*hour_factor, act, False]], 
                                                           columns = ["time", "actor", "update"]), ignore_index = True)

        else:             
            # for medical personal: Fill a maximum of three hours between two contacts
            filled_contacts = fill_space(contacts)
            hourly_array_full[i,:] = filled_contacts
            
            # create separate shift start and end data for dynam
            indices = np.where(filled_contacts == 1)[0]
            lens, pos = counter(indices)
            for curr_len, curr_pos in zip(lens, pos):
                # Add shiftstart and shiftend to df
                shift_collection = shift_collection.append(pd.DataFrame([[curr_pos*hour_factor, act, True]], 
                                                           columns = ["time", "actor", "update"]), ignore_index = True)
                shift_collection = shift_collection.append(pd.DataFrame([[(curr_pos+curr_len)*hour_factor, act, False]], 
                                                           columns = ["time", "actor", "update"]), ignore_index = True)

    # assume everybody is present initially who shows up within the first 6 hours of the study
    init_pres = np.sum(hourly_array_full[:,0:6], axis = 1)>0
    initial_present = pd.DataFrame({"actor": (actors["actor"]), "initial_pres": init_pres})
    
    return hourly_array_full, shift_collection, initial_present
    
    # fill hourly_array
    
agg_shift_array = time_agg(shift_array, axis = 1, agg_elems = 45*4)
fig, ax = plt.subplots(figsize=(26,15))
less_sparse_hm = sns.heatmap(agg_shift_array, yticklabels = actorstring)
fig.savefig("./Hourly_presence.jpg")


hourly_array_full, shift_collections, initial_present = shift_agg(shift_array, actors)
#hourly_array_full.to_csv("./hourly_shifts.csv", index = False)
shift_collections.to_csv("./shift_changes.csv", index = False)
initial_present.to_csv("./present_first_day.csv", index = False)


fig, ax = plt.subplots(figsize=(26,15))
less_sparse_hm = sns.heatmap(hourly_array_full, yticklabels = actorstring)
fig.savefig("./Hourly_shifts.jpg")