import json
import os
import re




class policy():



	global_nodes={}
	rc_map={}
	custom_outputs={}
	all_policies = {}
	nodes=[]

	def __init__(self,path_to_policies,org_id):

		self.path_to_policies=path_to_policies
		self.org_id=org_id
		

		self.read_policies()


	def read_policies(self):
		#print (self.path_to_policies)
		#print (self.org_id)
		#print (os.listdir(self.path_to_policies))
		policy.all_policies.update({'_'.join(p.split('_')[1:-1]).lower():self.path_to_policies+'/'+p for p in os.listdir(self.path_to_policies) if p.startswith(self.org_id+'_policy_') and p.endswith('json') })

	def list_policies(self):
		for pol in policy.all_policies.keys():
			print (pol)


	def get_rules_in_policy(self,policy_name):
		pol = json.load(open(policy.all_policies[policy_name],encoding='utf-8'))
		rule_set=pol['policyVersion']["rules"] + pol['policyVersion']["variables"] #change1			
		review_thresh=pol['policyVersion']["reviewStatus"][1]
		reject_thresh=pol['policyVersion']["reviewStatus"][0]
		print("review_threshold :"+str(review_thresh))
		print("reject_threshold :"+str(reject_thresh))

		return rule_set

	def build_graph(self,rules,policy_name):

		for r in rules:
			policy.nodes.append(node(r,policy_name))


	def display_node(self,rid):
		node_list=[rid]
		sub_graph_list=node_list+get_parents(node_list)
		sub_graph_nodes={n:policy.global_nodes[n] for n in sorted(sub_graph_list)}
		from graphviz import Digraph

		dot = Digraph(comment=rid)

		count=0
		
		for node in sub_graph_nodes:
			label1=str("\\n".join([str(policy.global_nodes[node].ID),policy.global_nodes[node].rulename,policy.global_nodes[node].ruletype]))
			label2=re.sub(">","GT",re.sub("<","LT","\\n ".join(policy.global_nodes[node].parameters)))
			label3="RiskWeight:"+str(policy.global_nodes[node].riskweight)
			#print label1,label2
			dot.node(str(policy.global_nodes[node].ID), "{ "+label1+" | "+label2+ "|"+ label3 +"}",shape='record',
							 fillcolor=get_color(policy.global_nodes[node].riskweight),style='filled')
			for parent in policy.global_nodes[node].parents:
				dot.edge(str(parent.ID), str(policy.global_nodes[node].ID))
			for d in policy.global_nodes[node].dependencies:
				dot.edge(str(policy.global_nodes[d].ID), str(policy.global_nodes[node].ID)) #change6
			count+=1

		return dot



class node():
		
	def __init__(self,rule,policy_name,parent=None):
			#print("node-policy",policy_name)
			self._rule=rule
			self.parents=[]
			self.children=[]
			self.policy_name=policy_name
			self.polcall=0
			self.rulename=rule['displayName']
			#print self.rulename
			if rule['ruleDisplayName']=="Call Policy": #change2
					self.polcall=1
			
			self.ruletype=rule['ruleDisplayName'] #change3
			self.ID= self.policy_name + '_' + rule['id'] #change7
			self.summary=rule.get('summary','')
			self.dependencies=[]
			self.riskweight=0
			if 'riskWeight' in rule:
					self.riskweight=rule['riskWeight']
			if parent is not None:
					self.parents.append(parent)
			if 'rules' in rule:
					for r in rule['rules']:
							self.add_children(r)
							
			try:
					k=[x['parameterName']+':'+tr_label(x['value']) for x in rule['parameters'] if x['parameterName']!="Dependencies"]
			except:
					print(rule['parameters'])
			
			self.parameters=[x['parameterName']+':'+tr_label(x['value']) for x in rule['parameters'] if x['parameterName']!="Dependencies"]
			self.parameters=sorted(self.parameters+['invert:'+str(rule['invert']),
																							'gen_rsn_cd:'+str(rule['generateReasonCode']),'summary:'+str(rule.get('summary',"NA"))])
			tmp = [x['value'] for x in rule['parameters'] if x['parameterName']=="Dependencies"]
			pc = [x['value'] for x in rule['parameters'] if x['parameterName']=="PolicyName"]
			if rule['ruleDisplayName']=='SL Execution Policy':
				pc = []
			
			# Each rule can output only one 
			custom_out = [x['value'] for x in rule['parameters'] if x['parameterName'] in ["Output_To","Output_Count_To"]]
			if len(custom_out)>0:
					print("custom output")
					policy.custom_outputs[custom_out[0]]=self.ID
					
			if len(tmp)>0: 
										self.dependencies= [self.policy_name + '_' + i for i in tmp[0].split('|')]  #change5
					
			custom_in = [policy.custom_outputs[x['value']] for x in rule['parameters'] if x['value'] in policy.custom_outputs and 
									x['parameterName'].startswith("Output")==False]
			if len(custom_in)>0:
					print(custom_in)
			self.dependencies+=custom_in
			if len(pc)>0:
					pname=re.sub(' ','_','policy_'+pc[0]).lower().replace("/","-")
					print (policy.all_policies[pname])
					pol1=json.load(open(policy.all_policies[pname]))
					rule_set1=pol1['policyVersion']["rules"] + pol1['policyVersion']["variables"] #change4
					for r1 in rule_set1: # TODO: rule names can clash accross policies
							
							if pname+'_'+r1['id'] not in policy.rc_map: #change9
									
									policy.nodes.append(node(r1,pname,self))
							else:
									policy.global_nodes[pname+'_'+r1['id']].parents.append(self) #change10
					
			policy.global_nodes[self.ID]=self
			
			policy.rc_map[self.ID]=rule['displayName']                                                                                                                                                
	
	def add_children(self,rule):
			self.children.append(node(rule,self.policy_name,self))

def tr_label(l):
		l=re.sub('[\\\]',"\\\\"+"\\\\",l)
		l=re.sub('[|]','\\|',l)
		l=re.sub('[{]','\\{',l)
		l=re.sub('[}]','\\}',l)
		return l

def get_color(x):
		if x<0:
				return "red"
		if x>0:
				return "green"
		return "white"

def get_parents(nlist):
		""" gets the nodes that are parents,dependencies recursively """
		#print "NLIST",nlist
		parent_list=[]
		
		for n in nlist:
				
				parent_list+=policy.global_nodes[n].dependencies+[x.ID for x in policy.global_nodes[n].parents]
				#print n,policy.global_nodes[n].dependencies+[x.ID for x in policy.global_nodes[n].parents]
		
		if len(parent_list)>0:
				parent_list+=get_parents(parent_list)
		return list(set(parent_list))


def get_dependents(nlist):
		node_set=set()
		
		for n in nlist:
				for g in policy.global_nodes:
						if n in policy.global_nodes[g].dependencies:
								node_set.add(g)
		return list(node_set)

def get_children(nlist):
		#print nlist
		if len(nlist)==0:
				return []
		child_list=get_dependents(nlist)
		for n in nlist:
				#print(n)
				child_list+=[x.ID for x in policy.global_nodes[n].children]
		
		child_list+=get_children(child_list)
		
		return list(set(child_list))


def check_params(params):
		import re
		lop= [re.compile(x) for x in ['local_attrib','custom_match','cc_bin_number[^_]','cc_bin_number$']]
		for p in lop:
				for par in params:
						if p.search(par) is not None:
								return True
		return False

def check_params2(params):
		import re
		lop= [re.compile(x) for x in ['condition_attrib','local_attrib','custom_match','cc_bin_number[^_]','cc_bin_number$']]
		for p in lop:
				for par in params:
						if p.search(par) is not None:
								return True
		return False
