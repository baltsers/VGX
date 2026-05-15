import Hierarchical
import Pattern
import AST
import os
import pickle
import multiprocessing
import difflib
import random
import sys
random.seed(3000)

def testPairs(pairs):
    succ=0
    fail=0
    '''
    targets=os.listdir('wild_funcs/wild_funcs_gen_devign')
    pos_targets=[]
    for i in range(len(targets)):
        if '_1.c' in targets[i]:
            pos_targets.append(targets[i][:-4])
    '''
    for i, pair in enumerate(pairs):
        #if i<7:
        #    continue
        if not pair.idx == "PatchDB_114688c526fe45f341d75ccd1d85473c3b08f7a7_vrend_create_vertex_elements_state":
            continue
        '''
        if not pair.idx in pos_targets:
            continue
        if os.path.exists('./generated/'+pair.idx+'_1.c'):
            continue
        '''
        before_ast = pair.getBeforeAst()
        after_ast = pair.getAfterAst()
        #import pdb
        #pdb.set_trace()
        rets = hierarchical_cluster.applyPattern(before_ast, pair.idx)
        #rets = [[],[],[]]
        new_asts = rets[1]
        subtree_root_ids = rets[2]
        patterns = rets[0]
        import pdb
        pdb.set_trace()
        '''
        for pattern_pre in hierarchical_cluster.pattern_prevalence_scores:
            pattern=pattern_pre[0]
            if 'fstrndup' in pattern.__str__():
                import pdb
                pdb.set_trace()
        '''
        flag = False
        before_nodes = before_ast.str2().split('\n')
        after_nodes = after_ast.str2().split('\n')
        diff = difflib.Differ()
        differences = list(diff.compare(before_nodes, after_nodes))
        del_flag = "del: false\n"
        add_flag = "add: false\n"
        for d in differences:
            if d[0] == '-':
                del_flag = "del: true\n"
            if d[0] == '+':
                add_flag = "add: true\n"

        '''
        test_asts = []
        for i, pattern in enumerate(patterns):
            for new_ast in new_asts[i]:
                test_asts.append(new_ast)
        #import pdb
        #pdb.set_trace()
        if not len(test_asts)-1 < 0:
            j = random.randint(0, len(test_asts)-1)
            test_ast = test_asts[j]
            if test_ast.getCode() == after_ast.getCode():
                flag = True
        '''
        breakflag = False
        for i, pattern in enumerate(patterns):
            #if not len(new_asts[i])-1 < 0:
                #j = random.randint(0, len(new_asts[i])-1)
            print(i,len(new_asts[i]))
            for j, new_ast in enumerate(new_asts[i]):
                #new_ast = new_asts[i][j]
                f=open('./generated/'+pair.idx+'_1.c','w')
                f.write(new_ast.getCode())
                f.close()
                
                if new_ast.getCode() == after_ast.getCode():
                    flag = True
                    print("    ", j)
                    breakflag = True
                    break
                #else:
                #    flag = False
                    #breakflag = True
                    #break
            if breakflag:
                break
        '''
        f=open('./generated/'+pair.idx+'_gth.c','w')
        f.write(after_ast.getCode())
        f.close()
        '''

        f=open('./generated/'+pair.idx+'_0.c','w')
        f.write(before_ast.getCode())
        f.close()
        if flag:
            print(pair.idx + "\nsuccess\n" + del_flag + add_flag)
        else:
            print(pair.idx + "\nfail\n" + del_flag + add_flag)
        if len(patterns)>0: #and 'kfree_skb ( skb ) ;' in patterns[0].getCode():
            #if not flag and len(patterns>0):
            if flag:
                succ+=1
            else:
                fail+=1
    print(succ,fail)


pairs = []
real_world = os.listdir('VulDatasets/real_world_final/test/')



#if os.path.exists('wild_funcs9.pkl'):
f = open(sys.argv[1],'rb')
pairs = pickle.load(f)
f.close()
'''
else:
    for file in real_world:
        if '.pkl' in file:
            f = open('VulDatasets/real_world_final/test/'+file, 'rb')
            parsed_data = pickle.load(f)
            for i,pair in enumerate(parsed_data):
                print(i)
                pairs.append(parsed_data[pair][0])
    f = open('real_world_test_pairs.pkl','wb')
    pickle.dump(pairs,f)
    f.close()
'''
#random.shuffle(pairs)
#pairs=pairs[10000:]

hierarchical_cluster = Hierarchical.HierarchicalCluster([])
if not os.path.exists('prevalence_spec4_mutated.pkl'):
    clusters = os.listdir('./clusters')
    for cluster in clusters:
        f = open('./clusters/'+cluster,'rb')
        hierarchical_cluster.hierarchical_nodes.extend(pickle.load(f))
        f.close()
#for i in range(90,108):
#    f = open('wild3'+str(i)+'.pkl','rb')
#    pairs = pickle.load(f)
#    f.close()
#    testPairs(pairs)

testPairs(pairs)

'''
pool = multiprocessing.Pool(9)
group_size = 2 #len(pairs)//100
groups = []
i = 0
while i < len(pairs):
    if i+group_size > len(pairs):
        groups.append(pairs[i:])
    else:
        groups.append(pairs[i:i+group_size])
    i += group_size
pool.map_async(testPairs, groups)
pool.close()
pool.join()
'''




