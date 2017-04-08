//package p6;

import static java.util.concurrent.TimeUnit.NANOSECONDS;

import java.util.TreeMap;

import crawler.readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile;
import crawler.readFile_load_wrapper_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile;

//calc_statistics
public class calc_statistics_no_authors_in_each_network {
	
	// calc_no_authors_in_each_network
	public static void calc_no_authors_in_each_network(
													   String baseFolder,
													   String inFile_all_token,
													   String inFile_AuthName_AuthID,
													   String in_TNC_substring_to_search,
													   String in_PNC_substring_to_search
													   ){
		int cnt_TNC=0;
		int cnt_PNC=0;
		try{
			//
			TreeMap<String, Integer> map_AuthName_AuthID=
								cleaning_and_convert.
								load_AuthName_and_Auth_id( inFile_AuthName_AuthID,	
														   true); //is_normalize_name
			
			TreeMap<Integer, String> map_lineno_Line=
			readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
			readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(inFile_all_token, 
																		-1, -1, "debug_label", false);
			
			//for each auth
			for(String curr_authName:map_AuthName_AuthID.keySet()){
				
				for(int lineno:map_lineno_Line.keySet()){
					String currLine=map_lineno_Line.get(lineno);
					
					if(currLine.indexOf(curr_authName) >=0 && currLine.indexOf(in_TNC_substring_to_search)>=0 
							){
						cnt_TNC++;
						break;
					}
					if(currLine.indexOf(curr_authName) >=0 && currLine.indexOf(in_PNC_substring_to_search)>=0 
							){
						cnt_PNC++;
						break;
					}
					
				}
				
			}
			
			System.out.println("TNC="+cnt_TNC+" PNC="+cnt_PNC);
			
		}
		catch(Exception e){
			e.printStackTrace();
		}
		
		
		
	}
	
	// main
	public static void main(String[] args) {
		long t0 = System.nanoTime();
		   String baseFolder="/Users/lenin/Downloads/p6/merged/dummy.mergeall/ds8/";
		   String inFile_all_token=baseFolder+"mergedall-2015-T8-10-Topics-all-tokens.txt";
		   String inFile_AuthName_AuthID=baseFolder+"authName_AND_auth_id_d.txt";
		   String in_TNC_substring_to_search="-trad-";
		   String in_PNC_substring_to_search="poli";
		   
			//calc_no_authors_in_each_network
			calc_no_authors_in_each_network(baseFolder,
											inFile_all_token,
											inFile_AuthName_AuthID,
											in_TNC_substring_to_search,
											in_PNC_substring_to_search
										    );
		System.out.println("Time Taken:" + NANOSECONDS.toSeconds(System.nanoTime() - t0) +" seconds; "
	   				  + (NANOSECONDS.toSeconds(System.nanoTime() - t0))/60 +" minutes");
	}
	 

}
