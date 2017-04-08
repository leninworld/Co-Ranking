

import java.io.File;

import java.io.FileWriter;
import java.io.IOException;
import java.util.TreeMap;

import crawler.readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile;

public class match_by_set_of_string_onEachLine_write_2_individualFile {
	
	//match_by_set_of_string_onEachLine_write_2_individualFile
	public static void match_by_set_of_string_onEachLine_write_2_individualFile(
																		String baseFolder,
																		String inputFile,
																		String query_CSV,
																		int	   index_token_to_Verify_eachLine_token,
																		int    index_of_token_having_primarykey,
																		int    index_of_token_having_authName,
																		int    top_N_to_be_matched_for_eachQuery,
																		String inFile10_dirty_authNames,
																		String output_all_matched_query_noDup
																		){
		 
		FileWriter writer=null;
		
		FileWriter writerDebug=null;
		FileWriter writer_output_all_matched_query_noDup=null;
		String []query=null;
		
		TreeMap<String, Integer> map_query_countOFtopNwroteSoFar=new TreeMap<String, Integer>();
		TreeMap<String, String> map_alreadyWrote_URLalreadyWROTE_in_asPartOFanyONEquery=new TreeMap<String, String>();
		
		try {
			query=query_CSV.split(",");
			int query_length=query.length;
			//
			writerDebug=new FileWriter(new File(baseFolder+"debug_individalquery.txt"));
			writer_output_all_matched_query_noDup=new FileWriter(new File(output_all_matched_query_noDup));
			
			//map_inputFile_only
			TreeMap<Integer,String> map_eachLine_dirty_authNames=
					  readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
								 .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
										 												 inFile10_dirty_authNames, 
																					 	 -1, //startline, 
																						 -1, //endline,
																						 " reloading..", //debug_label
																						 false //isPrintSOP
																						 );
			
			
			//map_inputFile_only
			TreeMap<Integer,String> map_inputFile_seq_line=
					  readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
								 .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
										 												 inputFile, 
																					 	 -1, //startline, 
																						 -1, //endline,
																						 " reloading..", //debug_label
																						 false //isPrintSOP
																						 );
			
			int total_doc=map_inputFile_seq_line.size();
			
			int c=0;
			////
			while(c<query_length){
				String curr_query=query[c];
				String file=baseFolder+curr_query.replace(" ", "") +".txt";
				
				 writer=new FileWriter(new File(file)); // ,true);
				
			for( int seq:map_inputFile_seq_line.keySet()){
				
				System.out.println("seq:"+seq+" out of total_doc="+total_doc);
				
				String orig_eachLine=map_inputFile_seq_line.get(seq);
				String []s=orig_eachLine.split("!!!");
				
				String currLine_authName=s[index_of_token_having_authName-1].toLowerCase();
				String currLine_primaryKEY=s[index_of_token_having_primarykey-1];
				currLine_primaryKEY=currLine_primaryKEY.replace("https", "http");
				
				//clean
				currLine_authName=currLine_authName.replace("from: ", "").replace("from:", "");
				
				// skip if it has dirty author name such as bbc
					if(   p6_MAIN_Work.is_given_author_a_dirtyAuthorName(currLine_authName, map_eachLine_dirty_authNames, orig_eachLine) ){
						
						if(seq %200000 ==0){
							writerDebug.append("dirtyAuthName:!!!"+ currLine_authName  +" line->"+orig_eachLine+"\n");
							writerDebug.flush();
						}
						continue;
					}
				
					if(s.length<index_token_to_Verify_eachLine_token){
						writerDebug.append("wrong no of tokens:"+orig_eachLine+"\n");
						writerDebug.flush();
					}
				 
					int index=orig_eachLine.toLowerCase().indexOf(curr_query.toLowerCase());
					
					//System.out.println("seq:"+seq+" out of total_doc="+total_doc+" index:"+index+" file:"+file +"s.length:"+s.length);
					
					// 
					if(index >=0 ){
						 
						if(!map_query_countOFtopNwroteSoFar.containsKey(curr_query)){
							
							 if(s.length==index_token_to_Verify_eachLine_token){ // NO TOKENS verified
								
								if(map_alreadyWrote_URLalreadyWROTE_in_asPartOFanyONEquery.containsKey(currLine_primaryKEY)){
									continue;
								}
								 
								writer.append(orig_eachLine+"\n");
								writer.flush();
								
								map_alreadyWrote_URLalreadyWROTE_in_asPartOFanyONEquery.put(currLine_primaryKEY , "");
								
								// COUNTING on query
//								if(map_query_countOFtopNwroteSoFar.containsKey(curr_query)){
//									int cnt=map_query_countOFtopNwroteSoFar.get(curr_query);
//									cnt=cnt+1;
//									map_query_countOFtopNwroteSoFar.put(curr_query , cnt);
//								}
//								else{
									map_query_countOFtopNwroteSoFar.put(curr_query, 1);
//								}
								
							 }
							 else{
								 System.out.println("violates no tokens");
							 }
						}
						else{
						
							//write only TOP N match for each query.
							if(map_query_countOFtopNwroteSoFar.get(curr_query) <= top_N_to_be_matched_for_eachQuery){
								
								if(s.length==index_token_to_Verify_eachLine_token){ // NO TOKENS verified
									
									if(map_alreadyWrote_URLalreadyWROTE_in_asPartOFanyONEquery.containsKey(currLine_primaryKEY)){
										continue;
									}
									
									writer.append(orig_eachLine+"\n");
									writer.flush();
									
									
									map_alreadyWrote_URLalreadyWROTE_in_asPartOFanyONEquery.put(currLine_primaryKEY , "");
									
									// COUNTING on query
									if(map_query_countOFtopNwroteSoFar.containsKey(curr_query)){
										int cnt=map_query_countOFtopNwroteSoFar.get(curr_query);
										cnt=cnt+1;
										map_query_countOFtopNwroteSoFar.put(curr_query , cnt);
									}
									else{
										map_query_countOFtopNwroteSoFar.put(curr_query, 1);
									}
									
								}
								else{
									 System.out.println("violates no tokens");
								 }
							}
							
						}
						 
					}
					

//					writer.close();
				}
				
				if(writer!=null)
					writer.close();
			
				c++;
				writer.close();
			}
			
			TreeMap<String, String> map_uniqueLinesOFallMATCHEDquery_as_KEY=new TreeMap<String, String>();
			TreeMap<String, String> map_alreadyPRIMARYKEY_as_KEY=new TreeMap<String, String>();
			c=0;
			//reach matched query already wrote and merge into one without duplicates..
			while(c<query_length){
				String curr_query=query[c];
				String file=baseFolder+curr_query.replace(" ", "") +".txt";
				 
				System.out.println("file:"+file);
				
				TreeMap<Integer,String> map_currQueryMATCHEDfile_eachLine=
						  readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
									 .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
											 												file, 
																						 	 -1, //startline, 
																							 -1, //endline,
																							 " reloading..", //debug_label
																							 false //isPrintSOP
																							 );

				System.out.println("loaded:"+map_currQueryMATCHEDfile_eachLine.size());
				for(int seq_:map_currQueryMATCHEDfile_eachLine.keySet() ){
					String currLine=map_currQueryMATCHEDfile_eachLine.get(seq_);
					String []s=currLine.split("!!!");
					String curr_primarykey=s[index_of_token_having_primarykey-1];
					// 
					curr_primarykey=curr_primarykey.replace("https", "http");
					
					if(!map_alreadyPRIMARYKEY_as_KEY.containsKey(curr_primarykey))
						map_uniqueLinesOFallMATCHEDquery_as_KEY.put(curr_primarykey, currLine);
					
					map_alreadyPRIMARYKEY_as_KEY.put(curr_primarykey, "");
				}
				///
				writerDebug.append("each query: "+curr_query+" file:"+file+" loaded:"+map_currQueryMATCHEDfile_eachLine.size()+
								" map_alreadyPRIMARYKEY_as_KEY:"+map_alreadyPRIMARYKEY_as_KEY.size()+"\n");
				writerDebug.flush();
				
				System.out.println("each query: "+curr_query+" file:"+file+" loaded:"+map_currQueryMATCHEDfile_eachLine.size()+
									" map_alreadyPRIMARYKEY_as_KEY:"+map_alreadyPRIMARYKEY_as_KEY.size()+"\n");
				c++;
			}
			
			//WRITE unique lines to 20000.txt
			for(String currPrimaryKey:map_uniqueLinesOFallMATCHEDquery_as_KEY.keySet()){
				writer_output_all_matched_query_noDup.append(map_uniqueLinesOFallMATCHEDquery_as_KEY.get(currPrimaryKey)+"\n");
				writer_output_all_matched_query_noDup.flush();
			}
			
			System.out.println("map_query_countOFtopNwroteSoFar:"+map_query_countOFtopNwroteSoFar);
			System.out.println("map_alreadyPRIMARYKEY_as_KEY:"+map_alreadyPRIMARYKEY_as_KEY.size());
			
		} catch (Exception e) {
			e.printStackTrace();
			// TODO: handle exception
		}
		
	}
	
	// main
    public static void main (String[] args) throws IOException {
    	
        String  baseFolder="/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds11/pick20000articles/";
        String inputFile=baseFolder+"mergedall-2016-T10-10-Topics-all-tokens.txt_ONLY_ENGLISH.txt_removed_NOISE.txt_REMOVE_STOPWORDS_FINAL.txt_MATCHED.txt_date_SORTED.txt";
        String query_csv=" india ,syria,britain,zika,boko haram,climate change,cancer,election,crime,trafficking";
        int top_N_to_be_matched_for_eachQuery=2000;
        int index_token_to_Verify_eachLine_token=9;
        int index_of_token_having_primarykey=2;
        int index_of_token_having_authName=6;
        
        String inFile10_dirty_authNames=baseFolder+"";
        String output_all_matched_query_noDup=baseFolder+"20000.txt";
        
        //NOTE: usually we use "sort_by_datetime_for_extracted_CSV_news_articles" and we sort a file by date (a token in a file)..
        //then the sorted OUTPUT file is given as input to this class match_by_set_of_string_onEachLine_write_2_individualFile. 
        
        // match_by_set_of_string_onEachLine_write_2_individualFile
        match_by_set_of_string_onEachLine_write_2_individualFile(
        														baseFolder,
        														inputFile,
        														query_csv,
        														index_token_to_Verify_eachLine_token,
        														index_of_token_having_primarykey,
        														index_of_token_having_authName,
        														top_N_to_be_matched_for_eachQuery,
        														inFile10_dirty_authNames,
        														output_all_matched_query_noDup
        													);

        
      
        
    }
	

}
