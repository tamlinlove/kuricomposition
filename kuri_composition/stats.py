import numpy as np

baseline_1 = [{'success':True,'num_actions': 12, 'total_time': 12.808843612670898, 'final_distance': 43.57177985806869},
{'success':True,'num_actions': 12, 'total_time': 12.816325426101685, 'final_distance': 20.396078054371138},
{'success':False,'num_actions': 12, 'total_time': 12.81595516204834, 'final_distance': 68.65311354920475},
{'success':True,'num_actions': 12, 'total_time': 12.784815549850464, 'final_distance': 52.376044142336674},
{'success':False,'num_actions': 12, 'total_time': 12.81651496887207, 'final_distance': 56.029010342857205}]

baseline_2 = [{'success':True,'num_actions': 12, 'error_time': 228.38506317138672, 'num_corrections': 214, 'total_time': 241.44009804725647, 'final_distance': 1.5},
{'success':True,'num_actions': 12, 'error_time': 201.79487085342407, 'num_corrections': 189, 'total_time': 214.7363841533661, 'final_distance': 8.0156097709407},
{'success':True,'num_actions': 12, 'error_time': 195.41853642463684, 'num_corrections': 183, 'total_time': 208.43204545974731, 'final_distance': 3.2015621187164243},
{'success':True,'num_actions': 12, 'error_time': 122.8998212814331, 'num_corrections': 115, 'total_time': 135.8882074356079, 'final_distance': 6.670832032063167},
{'success':True,'num_actions': 12, 'error_time': 185.647944688797, 'num_corrections': 174, 'total_time': 198.6777856349945, 'final_distance': 8.54400374531753}]

value_1 = [{'success':True,'num_actions': 12, 'error_time': 176.30015182495117, 'num_corrections': 165, 'total_time': 189.32792615890503, 'final_distance': 1.118033988749895},
{'success':True,'num_actions': 12, 'error_time': 195.1879961490631, 'num_corrections': 183, 'total_time': 208.3401448726654, 'final_distance': 8.139410298049853},
{'success':True,'num_actions': 12, 'error_time': 157.81962037086487, 'num_corrections': 143, 'total_time': 170.8070764541626, 'final_distance': 2.23606797749979},
{'success':True,'num_actions': 12, 'error_time': 188.96915459632874, 'num_corrections': 177, 'total_time': 201.93513441085815, 'final_distance': 2.5},
{'success':True,'num_actions': 12, 'error_time': 245.05291557312012, 'num_corrections': 229, 'total_time': 258.20259499549866, 'final_distance': 7.762087348130012}]

baseline_1_door = [{'success':False,'num_actions': 18, 'total_time': 19.214961051940918, 'final_distance': 562.878317223181},
{'success':False,'num_actions': 18, 'total_time': 19.222931146621704, 'final_distance': 145.7737973711325},
{'success':False,'num_actions': 18, 'total_time': 19.183568716049194, 'final_distance': 70.657625207758},
{'success':False,'num_actions': 18, 'total_time': 19.185149908065796, 'final_distance': 34.311076928595526} ,
{'success':False,'num_actions': 18, 'total_time': 19.248070240020752, 'final_distance': 134.67832045284794},]

def get_stats(name,res):
    times = []
    error_times = []
    distances = []
    num_corrects = []
    successes = 0
    error = False

    print(name)
    for run in res:
        if run["success"]:
            successes += 1
            times.append(run['total_time'])
            distances.append(run['final_distance'])
            if 'error_time' in run.keys():
                error = True
                error_times.append(run['error_time']/run['total_time'])
                num_corrects.append(run['num_corrections'])
    
    print("Successes: {}/{}".format(successes,len(res)))
    if successes > 0:
        print("Total Time: {} mean, {} std".format(np.mean(times),np.std(times)))
        print("Distance: {} mean, {} std".format(np.mean(distances),np.std(distances)))
        if error:
            print("Error Time: {} mean, {} std".format(np.mean(error_times),np.std(error_times)))
            print("Number of Corrections: {} mean, {} std".format(np.mean(num_corrects),np.std(num_corrects)))
    else:
        print("No successful runs")

if __name__ == "__main__":
    #get_stats("Policy, No Correct",baseline_1)
    #get_stats("Policy, Correct",baseline_2)
    #get_stats("VFC Policy, Correct",value_1)
    get_stats("Policy, No Correct, Door",baseline_1_door)

