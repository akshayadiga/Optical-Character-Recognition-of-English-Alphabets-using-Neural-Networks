import neurolab as nl

def toProb(letter):
	l=[]
	int_letter=ord(letter);
	pos=int_letter-97
	
	for i in range(26):
		if(i==pos):
			l.append(1)
		else:
			l.append(0)
	
	return l	


def main():
	# X = matrix of m input examples and n features

	f=open("letter.data","r")

	X=[]
	Y=[]
	count=0
	for line in f:
		vector=line.strip().split()
		in_vec=vector[6:]
		out_vec=vector[1]
		in_vec=[int(i) for i in in_vec]
		#out_vec=[int(i) for i in out_vec]

		X.append(in_vec)
		Y.append(out_vec)
		count=count+1
		if(count==800):
			break
		#X=numpy.matrix(X)

	f.close()	


	# Y = matrix of m output vectors, where each output vector has k output units


	#Y=numpy.matrix(Y)
	#print X
	#print Y

	Y=[toProb(i) for i in Y]
	
	
	
	
	net = nl.net.newff([[0, 1]]*128, [20, 26],transf=[nl.trans.TanSig(),nl.trans.SoftMax()])
	
	net.train(X, Y, epochs=20, show=1, goal=0.02)
	
	#z=net.sim([X[1]])
	#print z
	
	
	
	
	
	f=open("letter.data","r")
	X=[]
	Y=[]
	
	count=0
	for line in f:
		if(count<800):
			count=count+1
			continue
		vector=line.strip().split()
		in_vec=vector[6:]
		out_vec=vector[1]
		in_vec=[int(i) for i in in_vec]
		#out_vec=[int(i) for i in out_vec]

		X.append(in_vec)
		Y.append(out_vec)
		count=count+1
		if(count==1000):
			break

	z=net.sim(X)
	bit_let_pair=zip(X,Y)
	
	b=[i for p,i in bit_let_pair]
	
	correct=0
	incorrect=0
	let_predict=[]
	###change each index to appropriate letter#####
	for i in z:
		probs =i
		prob_letter=max(probs)
		for j in range(26):
		    	if(probs[j]==prob_letter):
		    		prob_pos=j
	    
		prob_pos+=97

		let_predict.append(chr(prob_pos))
	
	#print(let_predict)
	#print(b)
	
	
	################################
	for i in range(len(let_predict)):
		if(let_predict[i]==bit_let_pair[i][1]):
			correct+=1
		else:
			incorrect+=1	
		
	
	efficiency=correct/(float(correct+incorrect))
	print (efficiency*100),"%"

	#e = net.train(input, output, show=1, epochs=100, goal=0.0001)
	
main()
