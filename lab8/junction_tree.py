
# min-fill
# eiminam variabile in ordinea crescatoare nr muchilor de care e nevoie pt eliminarea lor
# use np array, transposition for factors
# g = {}
# g['a'] = ['b']
# g['b'] = ['d']
# g['c'] = ['d', 'e']
# g['d'] = ['f', 'g']
# g['e'] = ['j']
# g['f'] = ['i']
# g['g'] = ['j']
# g['h'] = ['k']
# g['i'] = ['k']
# g['j'] = ['l']
# g['k'] = []
# g['l'] = []

def get_undirected(g):
    gt = {}
    for x in g:
        for son in g[x]:
            if son not in gt:
                gt[son] = []
            gt[son].append(x)
    for x in g:
        for son in g[x]:
            if x not in gt:
                gt[x] = []
            gt[x].append(son) 
    return gt

def get_gt(g):
    gt = {}
    for x in g:
        for son in g[x]:
            if son not in gt:
                gt[son] = []
            gt[son].append(x)
    return gt

def moralize(gu, g, gt):
    for x in gt:
        for fath1 in gt[x]:
            for fath2 in gt[x]:
                if fath1 != fath2:
                    if fath2 not in gu[fath1]:
                        gu[fath1].append(fath2)
                    if fath1 not in gu[fath2]:
                        gu[fath2].append(fath1)

def print_graph(gu):
    pass
    # for x in gu:
    #     print(x)
    #     print(gu[x])

def get_needed_edges(gu, eliminated_nodes):

    needed_edges = {}
    for x in gu:
        if x not in eliminated_nodes:
            needed_edges[x] = 0
   
    for x in gu:
        if x in eliminated_nodes: 
            continue

        for s1 in gu[x]:
            if s1 in eliminated_nodes:
                continue
            for s2 in gu[x]:
                if s2 in eliminated_nodes:
                    continue
                # no edge between s1 and s2
                if s1 != s2 and s1 not in gu[s2]:
                    needed_edges[x] += 1
    return needed_edges

def triangularize(gu):

    eliminated_nodes = []

    # to be identical with the one from pdf
    gu['d'].append('e')
    gu['e'].append('d')
    pass

    while True:
        # get elimination edges
        ne = get_needed_edges(gu, eliminated_nodes)
        # sort by value
        sort_nodes_added_edges = sorted(ne.items(), key=lambda x: x[1], reverse=False)
        #print(sort_nodes_added_edges)
        if sort_nodes_added_edges[-1][1] == 0:
            break
        
        (node_el, _ ) = sort_nodes_added_edges[0] 
        
        eliminated_nodes.append(node_el)
        for son1 in gu[node_el]:
            
            if son1 in eliminated_nodes:
                continue
            
            for son2 in gu[node_el]:
                if son2 in eliminated_nodes:
                    continue

                if son1 != son2 and son1 not in gu[son2]:
                    gu[son1].append(son2)
                    gu[son2].append(son1)

def bron_kerbosch(gu, R, P, X):
    # R = vertices in the current clique
    # P = vertices that can be added to the clique
    # X = vertices that should not be added to the clique
    # P == 0 is maximal clique, X is for duplicates
    cliques = []
    if len(P) == 0 and len(X) == 0:
        cliques.append(R)
    for v in P:
        cliques += bron_kerbosch(gu, R.union(set([v])), P.intersection(set(gu[v])), X.intersection(set(gu[v])))
        # move v from P tu X
        P = P.difference(set([v]))
        X = X.union(set([v]))
    return cliques

def get_bit_table(x, n):

    bit_list = []
    for i in range(n):
        if ((1 << i) & x) != 0:
            bit_list.append(1)
        else:
            bit_list.append(0)
    bit_list.reverse()
    return bit_list

def compute_cpt(v, parents, probs):
    cpt = [parents + [v]]
    n = len(parents)

    for i in range( (1 << n) ):
        vals = get_bit_table(i, n)
        cpt.append(vals + [1] + [probs[i]])
        cpt.append(vals + [0] + [1 - probs[i]])
    # for x in cpt:
    #     print(x)
    return cpt

def read_bayes_network(filename):

    g = {}
    cpts = {}
    with open(filename, 'r') as f:
        for line in f:
            v, parentsS, probs = line.split(';')
            v = v.strip().lower()
            parentsS = parentsS.split()
            parents = []
            for p in parentsS:
                parents.append(p.lower())
            probss = probs.strip().split(' ')
            probs = []
            for p in probss:
                probs.append(float(p))
        
            for p in parents:
                if p in g:
                    g[p].append(v)
                else:
                    g[p] = [v]
            if v not in g:
                g[v] = []

            cpts[v] = compute_cpt(v, parents, probs)
    return g, cpts
                

def construct_clique_graph(gu, cliques):
    cg = {}
    for c in cliques:
        cg[frozenset(c)] = []

    for i in range(len(cliques)):
        for j in range(i + 1, len(cliques)):
            c1 = cliques[i]
            c2 = cliques[j]
            inter = c1.intersection(c2)
            if len(inter) > 0:
                cg[frozenset(c1)].append( (frozenset(c2), len(inter)) )
                cg[frozenset(c2)].append( (frozenset(c1), len(inter)) ) 

    return cg

def union(e1, e2, ss):
    x = ss[e1]
    for v in ss:
        if ss[v] == x:
            ss[v] = ss[e2]
    
def compute_max_span_tree(cg):

    edges = []
    for node1 in cg:
        for t in cg[node1]:
            if (node1, t[0], t[1]) not in edges and (t[0], node1, t[1]) not in edges:
                edges.append((node1, t[0], t[1]))
    edges = sorted(edges, key=lambda e: e[2], reverse = True)
    
    ss = {}
    index = 1
    for key in cg:
        ss[key] = index
        index += 1

    polytree = []
    clique_nodes = []
    for e in edges:
        if ss[e[0]] != ss[e[1]]:
            polytree.append( (e[0], e[1], e[2]) )
            if e[0] not in clique_nodes:
                clique_nodes.append(e[0])
            if e[1] not in clique_nodes:
                clique_nodes.append(e[1])

            union(e[0], e[1], ss)
    return polytree, clique_nodes

def get_all_matches(cpt2, common_values):
    all_matches = []
    for x2 in cpt2[1:]:
        match = True
        for i in range(len(x2[:-1])):
            if cpt2[0][i] in common_values:
                if x2[i] != common_values[cpt2[0][i]]:
                    match = False
        if match == True:
            all_matches.append(x2)
    return all_matches

def mul(res_vars, cpt1, x1, cpt2, x2):

    res_prob = []
    for v in res_vars:
        found = False
        for i in range(len(cpt1)):
            if cpt1[i] == v:
                res_prob.append(x1[i])
                found = True
        if found == False:
            for i in range(len(cpt2)):
                if cpt2[i] == v:
                    res_prob.append(x2[i])
    res_prob.append(x1[-1] * x2[-1])
    return res_prob

# multiply two factors
def multiply_factors(cpt1, cpt2):
    if len(cpt1) == 0:
        return cpt2
    if len(cpt2) == 0:
        return cpt1
    res_cpt = []
    common_factors = []
    ufac1 = []
    ufac2 = []

    for x in cpt1[0]:
        if x in cpt2[0]:
            common_factors.append(x)

    for x in cpt1[0]:
        if x not in common_factors:
            ufac1.append(x)

    for x in cpt2[0]:
        if x not in common_factors:
            ufac1.append(x)

    # for x in cpt1:
    #     print(x)
    
    # for x in cpt2:
    #     print(x)

    res_vars = ufac1 + common_factors + ufac2
    res_cpt.append(res_vars)

    for x1 in cpt1[1:]:
        # find all entries that match common vars values of x1
        # dict with common vars values
        common_values = {}
        for i in range(len(x1[:-1])):
            v = cpt1[0][i]
            if v in common_factors:
                common_values[v] = x1[i]
        matched_cpt2 = get_all_matches(cpt2, common_values)
        for x2 in matched_cpt2:
            res_entry = mul(res_vars, cpt1[0], x1, cpt2[0], x2)
            #print(res_entry)
            res_cpt.append(res_entry)
    
    # for x in res_cpt:
    #     print(x)
    return res_cpt

# if e1 matches e2 except at position pos
def match_except(e1, e2, pos):
    for i in range(len(e1[:-1])):
        if i != pos and e1[i] != e2[i]:
            return False
    return True

# sum e1 and e2 leaving pos_v out
def sum_entries(e1, e2, pos_v):
    res_e = []
    for i in range(len(e1[:-1])):
        if i != pos_v:
            res_e.append(e1[i])

    res_e.append(e1[-1] + e2[-1])
    return res_e

# sum out cpt factor by v variable
def sum_out_single(cpt, v):
    pos_v = -1
    res_cpt = []
    res_vars = []

    for i in range(len(cpt[0])):
        if cpt[0][i] == v:
            pos_v = i
        else:
            res_vars.append(cpt[0][i])
    res_cpt.append(res_vars)

    for i in range(1, len(cpt)):
        for j in range(i + 1, len(cpt)):
            if match_except(cpt[i], cpt[j], pos_v) == True:
                res_cpt.append(sum_entries(cpt[i], cpt[j], pos_v))
    return res_cpt

# sum out vaars from cpt
def sum_out_many(cpt, vaars):
    res_cpt = cpt
    for v in cpt[0]:
        if v not in vaars:
            res_cpt = sum_out_single(res_cpt, v)
    return res_cpt

# unity cpt, all probs are 1
def compute_unity(vs):
    cpt = [vs]
    n = len(vs)

    for i in range( (1 << n) ):
        vals = get_bit_table(i, n)
        cpt.append(vals + [1])
    return cpt

def compute_factors_nodes(polytree, clique_nodes, cpts):
    
    assoc_nodes = {cnode: [] for cnode in clique_nodes}
    factors = {}
    for cpt in cpts:
        cpt_v = cpts[cpt]
        for cnode in clique_nodes:
            contains_all = True
            # find association between nodes in bayes net and cliques
            for v in cpt_v[0]:
                if v not in cnode:
                    contains_all = False
                    break
            if contains_all == True:
                assoc_nodes[cnode].append(cpt_v[0][-1])
                break
    #print(assoc_nodes) 
    #multiply_factors(cpts['d'], cpts['e'])      
    #multiply_factors( [], cpts['d'] )      
    #print(assoc_nodes)
    for cnode in clique_nodes:
        
        cpt_fac = compute_unity(list(cnode))
        #cpt_fac = []
        for node in assoc_nodes[cnode]:
            #print(node)
            cpt_fac = multiply_factors(cpts[node], cpt_fac)
        # print(cnode)
        # print(cpt_fac)
        # print(assoc_nodes[cnode], cnode)
        factors[cnode] = cpt_fac
    # u = compute_unity(['a', 'b', 'c'])
    # for x in u:
    #     print(x)
    #multiply_factors([], cpts['d'])
    return factors

def eliminate(e, f):
    res = []
    vs = []
    # keep only the entries that match the evidence
    res.append(f[0])
    for i in range(len(f[1:])):
        for j in range(len(f[i + 1][:-1])):
            if f[0][j] == e[0] and f[i + 1][j] == e[1]:
                res.append(f[i + 1])
    return res

def incorporate_evidence(query, factors):

    for f in factors:
        cpt = factors[f]
        res = cpt
        for e in query[1]:
            if e[0] in cpt[0]:
                res = eliminate(e, res)
        factors[f] = res

def fix_root(polytree, factors, clique_nodes):
    g = {}
    for cnode in clique_nodes:
        g[cnode] = {}
        g[cnode]['sons'] = []
        g[cnode]['parents'] = []
    
    visited = {v: False for v in clique_nodes}
    # choose the root
    visited[clique_nodes[0]] = True
    done = False

    while done == False:
        done = True
        for e in polytree:
            if visited[e[0]] == True and visited[e[1]] == False:
                g[e[0]]['sons'].append(e[1])
                g[e[1]]['parents'].append(e[0])
                visited[e[1]] = True
                done = False
            if visited[e[1]] == True and visited[e[0]] == False:
                g[e[1]]['sons'].append(e[0])
                g[e[0]]['parents'].append(e[1])
                visited[e[0]] = True
                done = False
    return clique_nodes[0], g

def common_vars(cpt1, cpt2):
    common_vars = []
    for x1 in cpt1[0]:
        if x1 in cpt2[0]:
            common_vars.append(x1)
    return common_vars

def belief_prop_up(g, current, factors):
    
    fs_recv = []
    if len(g[current]['parents']) > 0:
        parent = g[current]['parents'][0]
    else:
        parent = None
    
    for son in g[current]['sons']:
        fs_recv.append(belief_prop_up(g, son, factors))

    to_send_factor = factors[current]
    #print(to_send_factor)
    #print(fs_recv)
    # multiply the factors received
    for f in fs_recv:
        to_send_factor = multiply_factors(to_send_factor, f)
    # keep only the common vars
    
    if parent != None:
        to_send_factor = sum_out_many(to_send_factor, common_vars(factors[current], factors[parent]))
    return to_send_factor
    
if __name__ == "__main__":
    g, cpts = read_bayes_network('bnet')
    all_nodes = [key for key, v in g.items()]
    gu = get_undirected(g)
    gt = get_gt(g)
    moralize(gu, g, gt)
    triangularize(gu)
    maximal_cliques = bron_kerbosch(gu, set(), set(all_nodes), set())
    cg = construct_clique_graph(gu, maximal_cliques)
    
    polytree, clique_nodes = compute_max_span_tree(cg)
   
    factors = compute_factors_nodes(polytree, clique_nodes, cpts)
    query = [ ['b', 'd'], [('a', 0), ('c', 1)] ]
    #incorporate_evidence(query, factors)
    root, tree = fix_root(polytree, factors, clique_nodes)
    res_root = belief_prop_up(tree, root, factors)
    for r in res_root:
        print(r)


    # for c in maximal_cliques:
    #     print(c)
    #cg = read_bayes_network(g, 'bnet')
    
    #
    
    
