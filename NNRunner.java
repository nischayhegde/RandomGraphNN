import java.awt.*;
import javax.swing.*;
import java.util.*;

public class NNRunner {

	public static void main(String[] args) {
		//xor test
		FORESTNN nn = new FORESTNN(2, 1);
		double[][] xorinputs = {{1,0},{0,0},{0,1},{1,1}};
		double[][] xoroutputs = {{1},{0},{1},{0}};

		int epochs = 1000;
		double[] output;
		double loss;
		Displayer.display(nn);
		for(int i=0; i<epochs; i++) {
			loss=0;
			for(int j=0; j<xorinputs.length; j++) {
				output=nn.feedforward(xorinputs[j]);
				loss+=nn.backtrack(xoroutputs[j]);
			}
			System.out.println("epoch: "+(i+1)+",loss: "+ loss);
		}

		System.out.println("tests: ");
		System.out.println(nn.feedforward(xorinputs[0])[0]);
		System.out.println(nn.feedforward(xorinputs[1])[0]);
		System.out.println(nn.feedforward(xorinputs[2])[0]);
		System.out.println(nn.feedforward(xorinputs[3])[0]);
	}
}

class Displayer
{
	public static void display(FORESTNN nn)
	{
		JFrame j = new JFrame();  //JFrame is the window; window is a depricated class
		Renderizer m = new Renderizer(nn);
		j.setSize(m.getSize());
		j.add(m); //adds the panel to the frame so that the picture will be drawn
			      //use setContentPane() sometimes works better then just add b/c of greater efficiency.

		j.setVisible(true); //allows the frame to be shown.

		j.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); //makes the dialog box exit when you click the "x" button.
	}

}

class Renderizer extends JPanel 
{
	FORESTNN nn;
	int sizex=1000;
	int sizey=1000;
	public Renderizer(FORESTNN nn)
	{
		this.nn=nn;
		setSize(sizex, sizey);
		setVisible(true); //it's like calling the repaint method.
	}
	
	public void paintComponent(Graphics g)
	{	
		g.setColor(Color.BLACK);
		g.fillRect(0, 0, sizex, sizey);
	  	int inputindex=0;
	  	int outputindex=0;
		for(int i=0; i<nn.neuronslist.size(); i++) {
			if(nn.neuronslist.get(i).startpoint) {
				int x=30;
				int y=(((sizey-200)/(nn.inputsize))*inputindex)+100;
				g.setColor(Color.GREEN);
				g.fillOval(x, y, 10, 10);
				inputindex++;
				nn.neuronslist.get(i).x=x;
				nn.neuronslist.get(i).y=y;
			}
			else if(nn.neuronslist.get(i).endpoint) {
				int x=sizex-50;
				int y=(((sizey-200)/(nn.outputsize))*outputindex)+100;
				g.setColor(Color.RED);
				g.fillOval(x, y, 10, 10);
				outputindex++;
				nn.neuronslist.get(i).x=x;
				nn.neuronslist.get(i).y=y;
			}
			else if(nn.neuronslist.get(i).included) {
				int x=(int)(Math.random()*(sizex-100)+50);
				int y=(int)(Math.random()*(sizey-100)+50);
				g.setColor(Color.BLUE);
				g.fillOval(x, y, 10, 10);
				nn.neuronslist.get(i).x=x;
				nn.neuronslist.get(i).y=y;
			}
		}
		g.setColor(new Color(255, 255, 255, 28));
		for(Neuron n : nn.neuronslist) {
			if(n.included) {
				for(int id : n.outputsID) {
					if(nn.neuronslist.get(id).included) {
						g.drawLine(n.x+5, n.y+5, nn.neuronslist.get(id).x+5, nn.neuronslist.get(id).y+5);
					}
				}
			}
		}
	}

}



class FORESTNN {
	public ArrayList<Neuron> neuronslist = new ArrayList<Neuron>();
	private double[] preds;
	private double learningrate;
	public int inputsize;
	public int outputsize;

	public FORESTNN(int inputsize, int outputsize) {
		int hiddenneurons=(int)Math.pow((inputsize+outputsize)*inputsize*outputsize, 2);
		int numconnections=(int)(Math.pow(hiddenneurons, 2)/(inputsize*outputsize));
		initializer(inputsize, outputsize, numconnections, hiddenneurons, 0.05);
	}

	public FORESTNN(int inputsize, int outputsize, double learningrate) {
		int hiddenneurons=(int)Math.pow(inputsize*outputsize, 2);
		int numconnections=(int)(Math.pow(hiddenneurons, 2)/2);
		initializer(inputsize, outputsize, numconnections, hiddenneurons, learningrate);
	}

	public void initializer(int inputsize, int outputsize, int numconnections, int hiddenneurons, double learningrate) {
		System.out.println("initializing...");
		int index=0;
		this.learningrate=learningrate;
		this.inputsize=inputsize;
		this.outputsize=outputsize;
		directedDFS dfs = new directedDFS(neuronslist);

		for(int i=0; i<inputsize; i++) {
			neuronslist.add(new Inputneuron(index, false, true));
			index++;
		}

		for(int i=0; i<outputsize; i++) {
			neuronslist.add(new Neuron(index, true, false));
			index++;
		}

		for(int i=0; i<hiddenneurons; i++) {
			neuronslist.add(new Neuron(index, false, false));
			index++;
		}

		int connectioncount = 0;

		while(connectioncount<numconnections) {
			boolean cyclic=false;

			while(true) {
				int from;
				int to;

				while(true) {
					from=(int)(Math.random()*neuronslist.size());
					to=(int)(Math.random()*neuronslist.size());
					if(!(neuronslist.get(from).endpoint || neuronslist.get(to).startpoint) && (!neuronslist.get(from).outputsID.contains(to))) break;
				}

				neuronslist.get(from).outputsID.add(to);
				neuronslist.get(to).inputsID.add(from);

				for(int i=0; i<neuronslist.size(); i++) {
					dfs.dfsreset();
					if(dfs.searchcycles(neuronslist.get(i), 1)==true) {cyclic=true; break;}
				}

				if(cyclic) {
					neuronslist.get(from).outputsID.remove(neuronslist.get(from).outputsID.size()-1); 
					neuronslist.get(to).inputsID.remove(neuronslist.get(to).inputsID.size()-1); 
					cyclic=false;
					continue;
				}
				else{connectioncount++; System.out.println("no cycles found, connection fused"); break;}
			}

		}
		dfs.dfsreset();
		for(int i=0; i<neuronslist.size(); i++) {
			neuronslist.get(i).initialize();
			if(!(neuronslist.get(i).inputsID.size()>0 || neuronslist.get(i).startpoint)) 
				neuronslist.get(i).included=false;
		}

		System.out.println("Done!!!");

		for(int i=0; i<neuronslist.size(); i++) System.out.println(neuronslist.get(i));
		preds = new double[outputsize];
	}

	public double[] feedforward(double[] inputs) {
		for(int i=0; i<neuronslist.size(); i++) {
			if(neuronslist.get(i).startpoint) {
				neuronslist.get(i).value=inputs[i];
			}
		}

		int index = 0;

		for(int i=0; i<neuronslist.size(); i++) {
			if(neuronslist.get(i).endpoint) {
				preds[index]=neuronslist.get(i).feedforward(neuronslist);
				index++;
			}
		}

		return preds;
	}

	public double backtrack(double[] realouts) {

		double[] outputgrad = dmselossdpred(preds, realouts);
		double loss = mseloss(preds, realouts);

		int index = 0;

		for(int i=0; i<neuronslist.size(); i++) {
			if(neuronslist.get(i).endpoint) {
				neuronslist.get(i).backtrackneuron(neuronslist.get(i).backtracktanh(outputgrad[index]), learningrate, neuronslist);
				index++;
			}
		}

		return loss;
	}

	public double mseloss(double[] pred, double[] real) {
		double sum=0;
		for(int i=0; i<pred.length; i++) {
			sum+=Math.pow(pred[i]-real[i], 2);
		}
		return sum/pred.length;
	}

	public double[] dmselossdpred(double[] pred, double[] real) {
		double[] derivatives = new double[pred.length];
		for(int i=0; i<pred.length; i++) {
			derivatives[i] = (2*pred[i]-2*real[i])/real.length;
		}
		return derivatives;
	}
}


class directedDFS {
	Stack<Neuron> stack = new Stack<Neuron>();
	ArrayList<Neuron> neuronslist;
	public directedDFS(ArrayList<Neuron> neuronslist) {
		this.neuronslist=neuronslist;
	}

	public void dfsreset() {
		stack = new Stack<Neuron>();
		for(int i=0; i<neuronslist.size(); i++) {
			neuronslist.get(i).visited=-1;
		}
	}

	public boolean searchcycles(Neuron neuron, int searchnum) {
		if(searchnum==1) stack.push(neuron);
		neuron.visited=0;
		if(neuron.outputsID.size()>0) {
			int newconns=0;
			for(int connection : neuron.outputsID) {
				if(!(neuronslist.get(connection).visited==1)){
					stack.push(neuronslist.get(connection));
					newconns++;
					if(neuronslist.get(connection).visited==0) {System.out.println("cycle detected, connection aborted"); return true;}
				}
			}
			if(newconns==0) {neuron.visited=1; stack.pop();}
		}
		else {neuron.visited=1; stack.pop();}
		if(stack.size()==0) {return false;}
		return searchcycles(stack.lastElement(), searchnum+1);
	}
}

class Neuron {
	public ArrayList<Integer> inputsID = new ArrayList<Integer>();
	public ArrayList<Integer> outputsID = new ArrayList<Integer>();

	public boolean included=true;

	public int ID;

	private double weights[];
	private double bias;

	public boolean endpoint;
	public boolean startpoint;

	private double output;
	private double[] inputs;
	public int visited=-1;
	public double value;
	public int x;
	public int y;

	public Neuron(int ID, boolean endpoint, boolean startpoint) {
		this.ID = ID;
		this.endpoint=endpoint;
		this.startpoint=startpoint;
	}

	public void initialize() {
		bias=(Math.random()-0.5);
		weights=new double[inputsID.size()];
		for(int i=0; i<weights.length; i++) weights[i]=(Math.random()-0.5);
	}

	public double feedforward(ArrayList<Neuron> neuronslist) {
		double inputsum=0;
		inputs = new double[inputsID.size()];
		for(int i=0; i<inputsID.size(); i++) {
			if(neuronslist.get(inputsID.get(i)).included) {
				inputs[i]=neuronslist.get(inputsID.get(i)).feedforward(neuronslist);
				inputsum+=inputs[i]*weights[i];
			}

		}
		output=Math.tanh(inputsum+bias);
		return output;
	}

	public double backtracktanh(double dE_wrt_Y) {
		double dE_wrt_X;
		dE_wrt_X=(1-Math.pow(output,2))*dE_wrt_Y;
		return dE_wrt_X;
	}

	public void backtrackneuron(double dE_wrt_Y, double learningrate, ArrayList<Neuron> neuronslist) {
		double wsum=0;
		for(double weight : weights) {
			wsum+=weight;
		}
		double[] dE_wrt_dW=inputs.clone();
		double dE_wrt_dB=dE_wrt_Y;
		double dE_wrt_dX=wsum*dE_wrt_Y;

		bias-=learningrate*dE_wrt_dB;

		for(int i=0; i<weights.length; i++) {
			weights[i]-=learningrate*dE_wrt_dW[i]*dE_wrt_Y;
		}

		for(int i=0; i<inputsID.size(); i++) {
			if(neuronslist.get(inputsID.get(i)).included)
				neuronslist.get(inputsID.get(i)).backtrackneuron(neuronslist.get(inputsID.get(i)).backtracktanh(dE_wrt_dX), learningrate, neuronslist);
		}
	}

	public String toString() {
		String feedback;
		if(endpoint)
			feedback = "OutputNeuronID: "+ID+" Connected to: ";
		else 
			feedback = "HiddenNeuronID: "+ID+" Included: "+included+" Connected to: ";
		for(int ID : outputsID) feedback+= ID+", ";
		return feedback;
	}
}

class Inputneuron extends Neuron {

	public Inputneuron (int ID, boolean endpoint, boolean startpoint) {
		super(ID, endpoint, startpoint);
	}

	@Override
	public double feedforward(ArrayList neuronslist) {
		return value;
	}

	@Override
	public double backtracktanh(double dE_wrt_dY) {
		return 0;
	}

	@Override
	public void backtrackneuron(double dE_wrt_dY, double learningrate, ArrayList neuronlist) {
		;
	}

	@Override
	public String toString() {
		String feedback = "InputNeuronID: "+ID+" Included: "+included+" Connected to: ";
		for(int ID : outputsID) feedback+= ID+", ";
		return feedback;
	}

}