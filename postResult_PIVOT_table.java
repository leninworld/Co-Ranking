

import java.io.IOException;
import java.util.TreeMap;

public class postResult_PIVOT_table {

	
	
	
	// main
	public static void main(String[] args) throws IOException {
	
		//five lines are given
		//each line is a output from Flag=34 of ground_truth.java (each line = a particular "N" value) 
		
		//each first line ends with # 
		//each line has -> first three for TRAD (P@N,MAP@N,DCG@N), second 3 tokens for POLI (P@N,MAP,DCG@N)
		String top10_top50_in_eachRow=
						"0.3233333333333333,0.5001644640097017,0.6855386026837856,0.46666666666666673,0.6962703716651218,0.8447760697517409#"+
					    "0.3083333333333333,0.43940370854399186,0.6867943617089468,0.525,0.7112674035671079,0.860658087993475#"+
						"0.3444444444444445,0.5185316344141506,0.6829179197500964,0.5277777777777778,0.696469663533953,0.8532364679580698#"+
						"0.32916666666666666,0.5118653565423306,0.6834331609567607,0.45416666666666666,0.6854842411056336,0.8504319840990954#"+
						"0.3233333333333333,0.5001644640097017,0.6855386026837856,0.46666666666666673,0.6962703716651218,0.8447760697517409";
						 

		
		String output_eachLine_conc="";
		String [] arr_first_eachLine=top10_top50_in_eachRow.split("#");
		int max_cnt=arr_first_eachLine.length;int c=0;
		
		int number_of_tokens_in_eachFirstLines=5;
		int c2=0;
		
		TreeMap<Integer,String> map_SEQ_outputEACHLINE_toPrint=new TreeMap<Integer, String>();
		
		while(c2<=number_of_tokens_in_eachFirstLines){
			output_eachLine_conc="";
			c=0;
			//
			while(c<max_cnt){
				String curr_first_EachLine=arr_first_eachLine[c];
				String []arr_second_eachLine=curr_first_EachLine.split(",");
				int max_second_eachLine=arr_second_eachLine.length;
				int c3=0;
				//
				while(c3<max_second_eachLine){
//					System.out.println(c3+1+" "+c2);
					if(c3+1==c2 ){
						if(output_eachLine_conc.length()==0)
							output_eachLine_conc=arr_second_eachLine[c3];
						else
							output_eachLine_conc=output_eachLine_conc+","+arr_second_eachLine[c3];
					}
					
					c3++;
				}
				c++;
			}
			if(output_eachLine_conc.length()>0){
				System.out.println(output_eachLine_conc);
				int size=map_SEQ_outputEACHLINE_toPrint.size()+1;
				map_SEQ_outputEACHLINE_toPrint.put(size, output_eachLine_conc);
			}
			
			c2++;
		}
		
		//get the last token from each line
		 c2=0;
			
			while(c2<=number_of_tokens_in_eachFirstLines){
				output_eachLine_conc="";
				c=0;
				//
				while(c<max_cnt){
					String curr_first_EachLine=arr_first_eachLine[c];
					String []arr_second_eachLine=curr_first_EachLine.split(",");
					int max_second_eachLine=arr_second_eachLine.length;
					int c3=0;
 
						if(output_eachLine_conc.length()==0)
							output_eachLine_conc=arr_second_eachLine[5];//last token
						else
							output_eachLine_conc=output_eachLine_conc+","+arr_second_eachLine[5];
					  
					c++;
				}
				if(c==max_cnt){
					if(output_eachLine_conc.length()>0){
						System.out.println(output_eachLine_conc);
						int size=map_SEQ_outputEACHLINE_toPrint.size()+1;
						map_SEQ_outputEACHLINE_toPrint.put(size, output_eachLine_conc);
					}
				}
				c2++;
			}
		
			System.out.println("map_SEQ_outputEACHLINE_toPrint:"+map_SEQ_outputEACHLINE_toPrint);
			
			//print in  sequence 1,2,4,5,3,6
			
			System.out.println("--------------------");
			System.out.println( map_SEQ_outputEACHLINE_toPrint.get(1)+"\n"+
								map_SEQ_outputEACHLINE_toPrint.get(2)+"\n"+
								map_SEQ_outputEACHLINE_toPrint.get(4)+"\n"+
								map_SEQ_outputEACHLINE_toPrint.get(5)+"\n"+
								map_SEQ_outputEACHLINE_toPrint.get(3)+"\n"+
								map_SEQ_outputEACHLINE_toPrint.get(6)+"\n"
								);
			
			
	}
	
}
