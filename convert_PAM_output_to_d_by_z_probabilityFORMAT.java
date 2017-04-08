

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.TreeMap;

import crawler.readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile;

public class convert_PAM_output_to_d_by_z_probabilityFORMAT {

	// mallet_topic_PAM_output_convert_2_d_by_z_probability
	public static void mallet_topic_PAM_output_convert_2_d_by_z_probability(
																			String inputFile,
																			String outputFile
																			){
		
		try{
			
	    	  // Loading seq and Line 
			  TreeMap<Integer, String>  map_seq_inFile_TEMP=
					  readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
										.readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
																										 inputFile, 
																									 	 -1, //startline, 
																										 -1, //endline,
																										 " loading", //debug_label
																										 false //isPrintSOP
																										 );
			  
			  FileWriter writer=new FileWriter(new File(outputFile));
			  
			  //
			  writer.append(map_seq_inFile_TEMP.get(1)+ "\n");
			  writer.flush();
			  
			  // 
			  for(int seq:map_seq_inFile_TEMP.keySet()){
				  
				  if(seq==1) continue;
				  
				  String currLine=map_seq_inFile_TEMP.get(seq);
				   
				  int flag=1;
				  
				   String [] arr=currLine.split(" ");
				   String currDocID= arr[0];
				   
				   
				   if(flag==1){
					   try{
						   currLine=currLine.substring(0, currLine.indexOf(",")-1);
						   arr=currLine.split(" ");
						   currDocID= arr[0];
						   
						   //example super-topic:5 0.2982456140350877 0 0.21052631578947367 4 0.17543859649122806 1 0.12280701754385964 2 0.11403508771929824 3 0.07894736842105263 
						   writer.append(currDocID+" "+arr[2]+" "+arr[3] +"\n");
						   writer.append(currDocID+" "+arr[4]+" "+arr[5] +"\n");
						   writer.append(currDocID+" "+arr[6]+" "+arr[7] +"\n");
						   writer.append(currDocID+" "+arr[8]+" "+arr[9] +"\n");
						   writer.append(currDocID+" "+arr[10]+" "+arr[11] +"\n");
						   writer.append(currDocID+" "+arr[12]+" "+arr[13] +"\n");
						   writer.flush();
					   }
					   catch(Exception e){
						   System.out.println("ERROR: seq:"+seq+" "+arr.length+" currDocID:"+currDocID+" line:"+currLine);
					   }
				   }
				   
				   if(flag==2){
					   currLine= currLine.substring(  currLine.indexOf(","), currLine.length()).replace(", ", "").replace(",", ""); 
					   arr=currLine.split(" ");
					   
					   System.out.println("seq:"+seq+" "+arr.length+" currDocID:"+currDocID+" line:"+currLine);
					   String temp="";
					   try{
						   //example super-topic:5 0.2982456140350877 0 0.21052631578947367 4 0.17543859649122806 1 0.12280701754385964 2 0.11403508771929824 3 0.07894736842105263
						   
 
						   
						   writer.append(currDocID+" "+arr[0]+" "+arr[1] +"\n");
						   
						   
						   writer.append(currDocID+" "+arr[2]+" "+arr[3] +"\n");
						   writer.append(currDocID+" "+arr[4]+" "+arr[5] +"\n");
						   writer.append(currDocID+" "+arr[6]+" "+arr[7] +"\n");
						   writer.append(currDocID+" "+arr[8]+" "+arr[9] +"\n");
						   writer.append(currDocID+" "+arr[10]+" "+arr[11] +"\n");
						   writer.flush();
					   }
					   catch(Exception e){
						   System.out.println("ERROR: seq:"+seq+" "+arr.length+" currDocID:"+currDocID+" line:"+currLine);
					   }
				   
				   }
				   
			  }
			  
			
		}
		catch(Exception e){
			e.printStackTrace();
		}
		
	}
	
	// MAIN
	public static void main(String[] args) throws IOException {
		
		String baseFolder="/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/";
		String inputFile=baseFolder+"pam_OUTPUT_backup.txt";
		String outputFile=baseFolder+"PAM_d_by_z.txt";

		// mallet_topic_PAM_output_convert_2_d_by_z_probability
		mallet_topic_PAM_output_convert_2_d_by_z_probability(
														inputFile,
														outputFile
														);
		
	}
	
}
