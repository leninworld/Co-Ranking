/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */



import static java.util.concurrent.TimeUnit.NANOSECONDS;

import org.apache.commons.lang3.CharUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.poi.ss.formula.functions.Replace;
import crawler.readFile_eachLine_get_a_particular_token_calc_frequency;
import cern.colt.list.IntArrayList;
import convert.getKeyByValueInTreeMap;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;

import crawler.DocumentParser_AND_Cosine;
import crawler.LoadMultipleValueToMap2_AS_KEY_VALUE;
import crawler.clean_retain_only_alpha_numeric_characters;
import crawler.convert_doc_word_FREQ_file_to_doc_word_TFIDF;
import crawler.remove_commas_etc;
import crawler.sort_given_treemap;
import crawler.tf_idf_AND_cosine_main;
import crawler.readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile;
import crawler.removeUnicodeChar;
import crawler.splitCSVAuthors_to_atom;
import crawler.stopwords;

/**
 *
 * @author lenin
 */
public class p6_MAIN_Work {

	//normalize_duplicate_authID_applying_cross_multiAuthIDs
	private static void normalize_duplicate_authID_applying_cross_multiAuthIDs(
															String baseFolder,
															String outputFile_cross_multiAuthID_SORTED_addMINauthID,  //in
															String outFile_auth_id_doc_id_queryTopicRelated, //in
															String outFile_auth_id_doc_id_queryTopicRelated_applying_cross_multiAuthIDs //out
															) {
		
		// TODO Auto-generated method stub
		try {
			
			FileWriter writerDebug =new FileWriter(new File( baseFolder+"debug_apply_crossmultiauthid.txt" ));
			FileWriter writer =new FileWriter(new File(outFile_auth_id_doc_id_queryTopicRelated_applying_cross_multiAuthIDs));
			 
			
			  TreeMap<Integer, String> mapIS_cross_multiAuthID_SORTED_addMINauthID=
			   	        readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
			   	        .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
			   	        													  outputFile_cross_multiAuthID_SORTED_addMINauthID,
			   	        													  -1, -1
			   									            				  , " load  auth 1 id & doc id"
			   									            				  , false //isPrintSOP
			   									            				  );
			  
			  TreeMap<Integer, String> mapIS_auth_id_doc_id_queryTopicRelated=
			   	        readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
			   	        .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
			   	        													  outFile_auth_id_doc_id_queryTopicRelated,
			   	        													  -1, -1
			   									            				  , " load  auth 2 id & doc id"
			   									            				  , false //isPrintSOP
			   									            				  );
			  TreeMap<Integer, Integer> map_pair_mapping_authID_crossMultiauthID=new TreeMap<Integer, Integer>();
			  
			  // 
			  for(int seq:mapIS_cross_multiAuthID_SORTED_addMINauthID.keySet()){
				  if(seq==1) continue;
				  String line=mapIS_cross_multiAuthID_SORTED_addMINauthID.get(seq);
				  String []s=line.split("!!!");
				  
				  String ALL_authIDs=s[4];
				  String min_AUTHID=s[5];
				  System.out.println("ALL_authIDs:"+ALL_authIDs);
				  //
				  if( ALL_authIDs.indexOf(",") >=0  ){
					  writerDebug.append(", included ->"+ALL_authIDs +"\n");
					  writerDebug.flush();
					  //not equals means ALL_authIDs has CSV authIDs
					  if(	 !ALL_authIDs.equalsIgnoreCase(min_AUTHID)
//						  && Integer.valueOf( min_AUTHID ) != Integer.valueOf(ALL_authIDs)
							  ){
						  String [] s2=ALL_authIDs.split(",");
						  int cnt=0;
						  //
						  while(cnt<s2.length ){
							  if( Integer.valueOf( s2[cnt].trim() )!=Integer.valueOf( min_AUTHID.trim() ) ){
								  map_pair_mapping_authID_crossMultiauthID.put( Integer.valueOf( s2[cnt] )  , 
										  										Integer.valueOf( min_AUTHID ) 
										  									  );
							  }
								  
							  cnt++;
						  }  
					  } //if(!ALL_authIDs.equalsIgnoreCase(min_AUTHID)){
				  }
				  
			  } // END for(int seq:mapIS_cross_multiAuthID_SORTED_addMINauthID.keySet()){
			  
			  writer.append("authID!!!docID!!!query\n");
			  // mapIS_auth_id_doc_id_queryTopicRelated
			  for(int seq2:mapIS_auth_id_doc_id_queryTopicRelated.keySet()){
				  String line=mapIS_auth_id_doc_id_queryTopicRelated.get(seq2);
				  String [] s3=line.split("!!!");
				  int authID= Integer.valueOf(s3[0]);
				  int docID=  Integer.valueOf(s3[1]);
				  String query= (s3[2]);
				  // 
				  if( map_pair_mapping_authID_crossMultiauthID.containsKey(authID) ){
					  //min authID
					  writer.append(map_pair_mapping_authID_crossMultiauthID.get(authID)  +"!!!"+ docID+"!!!"+query+ "\n");
					  writer.flush();  
					  //
					  writerDebug.append("mappnig:"+ authID +"-->"+map_pair_mapping_authID_crossMultiauthID.get(authID) +"\n");
					  writerDebug.flush();
				  }
				  else{
					  writer.append( authID+"!!!"+ docID+"!!!"+query+ "\n");
					  writer.flush();
				  }
			  }
			  
			  //
			  writerDebug.append("map_pair_mapping_authID_crossMultiauthID:"+map_pair_mapping_authID_crossMultiauthID);
			  writerDebug.flush();
			  
		} catch (Exception e) {
			e.printStackTrace();
			// TODO: handle exception
		}
		
	}
	
	//remove_whiteSpace_in_voc
	private static void remove_whiteSpace_in_voc(String outFile_for_vocabulary, String OUTPUT_no_whitespace) {
		// TODO Auto-generated method stub
		TreeMap<String,String> map_UniqVocID_Word_LOADED=
 				LoadMultipleValueToMap2_AS_KEY_VALUE.LoadMultipleValueToMap2_AS_KEY_VALUE(outFile_for_vocabulary, "!!!",
 																							"1,2",
 																							"dummy", //append_suffix_at_each_line
 																							false,   //is_have_only_alphanumeric
 																							false);
     	//
     	try {
			FileWriter writer_vocnospa=new FileWriter(new File(OUTPUT_no_whitespace));
			//ITERATE and WHITE SPACE
	     	for(String wordid:map_UniqVocID_Word_LOADED.keySet()){
	     		//
	     		String 	word_no_whitespace=map_UniqVocID_Word_LOADED.get(wordid);
	     				 
	     		String t=Replace_WhiteSpaceSpecialChar_with_3Hash(word_no_whitespace);
 	     		String [] s = t.split("###");
	     		
	     		System.out.println(  "s.len:"+s.length +"--"+word_no_whitespace+"---"+t);
	     		 //retain only alpha numeric
//	     		word_no_whitespace= 
//		     					  	clean_retain_only_alpha_numeric_characters.
//		     					  	clean_retain_only_alpha_numeric_characters(word_no_whitespace, false);
	     		
	     		writer_vocnospa.append(wordid+"!!!"+word_no_whitespace+"\n");
	     		writer_vocnospa.flush();
	     	}
	     	writer_vocnospa.close();
	     	System.out.println("fil:"+outFile_for_vocabulary);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

     	
        /////////////////////remove WHITE SPACE
     	 

     	
	}
	
	private final static Pattern LTRIM = Pattern.compile("^\\s+");
	private final static Pattern RTRIM = Pattern.compile("\\s+$");

	public static int CharToASCII(final char character){
		return (int)character;
	}
	
	//Replace_WhiteSpaceSpecialChar_with_3Dollar
	public static String Replace_WhiteSpaceSpecialChar_with_3Hash(String s) {
		 
		//2. Without Using replaceAll() Method
		 
        char[] strArray = s.toCharArray();
 
        StringBuffer sb = new StringBuffer();
 
        for (int i = 0; i < strArray.length; i++)
        { 
        	// http://www.mkyong.com/java/how-to-convert-character-to-ascii-in-java/
            if( (strArray[i] != ' '  ) && (strArray[i] != '\t') && ( CharToASCII(strArray[i]) < 124
            		|| CharToASCII(strArray[i]) == 156 // pound
            		)
            	)
            {
            	//System.out.println("strArray[i]:"+strArray[i] +" "+  CharToASCII(strArray[i] ) );
                sb.append(strArray[i]);
            }
            else{
            	sb.append("###");
            }
        }
 
//        System.out.println("m2:"+sb); 
		
	    return sb.toString();
	}
    //rtrim
	public static String rtrim(String s) {
	    return RTRIM.matcher(s).replaceAll("");
	}
	
	//p6_normalize_auth_name
	public static String p6_normalize_auth_name(String authorName){
		
		return authorName.replace("auth:from: :from:", "auth:from: ")
			.replace("auth:from:  ", "auth:from: ").replace("auth:from::from:", "auth:from:")
			.replace("auth:from:","").replace("auth name:", "")
			.replace("  ", " ");
		
	}
	//remove_stopWordID_from_authID_wordID_freq
	private static void remove_stopWordID_from_authID_wordID_freq(
														String baseFolder,
														String outFile_for_author_id_and_word_id_with_frequency,
														String outFile_for_vocabulary,
														String outputFile_removedStopWordID
														) {
		// TODO Auto-generated method stub
		
		FileWriter writer_output=null;
		TreeMap<Integer, String > mapIS_eachLINE_docID_WordID_Frequency=new TreeMap<Integer, String>();
		TreeMap<String, Integer> map_StopWord_WordID=new TreeMap<String, Integer>();
		String debug_File=baseFolder+"debug_removeStopWordID.txt";
		FileWriter writerDebug=null; String line="";
    	stopwords  stopwords_ =new stopwords(); 
    	TreeMap<String,String> map_UniqVocID_Word_LOADED=
				LoadMultipleValueToMap2_AS_KEY_VALUE.LoadMultipleValueToMap2_AS_KEY_VALUE(outFile_for_vocabulary, "!!!",
																							"1,2",
																							"dummy",//append_suffix_at_each_line
																							true, //is_have_only_alphanumeric
																							false);
    	TreeMap<String,String> map_Word_UniqVocID_LOADED=
			LoadMultipleValueToMap2_AS_KEY_VALUE.LoadMultipleValueToMap2_AS_KEY_VALUE_REVERSE(outFile_for_vocabulary, "!!!",
																							  true, //is_have_only_alphanumeric
																							  false);
    	
    	TreeMap<String, String >  map_StopWord_dummy=stopwords_.stopwords();
    	
	  
//		BufferedReader br=new BufferedReader(new FileReader(new File(inFile_1_for_each_doc_word_id_AND_FREQ)));
		try {
			
			writer_output=new FileWriter(new File(outputFile_removedStopWordID ));
			writerDebug=new FileWriter(new File(debug_File));
			
			writerDebug.append("map_Word_UniqVocID_LOADED:"+map_Word_UniqVocID_LOADED+"\n");
			writerDebug.append("----------------------------------------------------------------");
	    	writerDebug.flush();
			
			
	       	///get wordID of stopword
	    	for(String stopword:map_StopWord_dummy.keySet()){
	    		System.out.println("stopword:"+stopword +" "+map_Word_UniqVocID_LOADED.containsKey(stopword));
	    		stopword=stopword.replaceAll("[\u0000-\u001f]", "");
	    		try {
					writerDebug.append("stopword:"+stopword +" 1:"+map_Word_UniqVocID_LOADED.containsKey(stopword)
										+" 2:"+map_Word_UniqVocID_LOADED.containsKey(stopword+" ")
										+" 3:"+map_Word_UniqVocID_LOADED.containsKey(" "+stopword)
										+" 4:"+map_Word_UniqVocID_LOADED.containsKey(" "+stopword+" ")
										+"\n");
					writerDebug.flush();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
	    		
	    		
	    		if(map_Word_UniqVocID_LOADED.containsKey(stopword)
	    			||map_Word_UniqVocID_LOADED.containsKey(stopword+" ")
	    			||map_Word_UniqVocID_LOADED.containsKey(" "+stopword)
	    			||map_Word_UniqVocID_LOADED.containsKey(" "+stopword+" ")
	    				){
	    			map_StopWord_WordID.put(stopword.trim() , Integer.valueOf(map_Word_UniqVocID_LOADED.get(stopword)) );
	    		}
	    	}
	    	
			System.out.println("loading...");
				///
			  mapIS_eachLINE_docID_WordID_Frequency=
			   	        readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
			   	        .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
			   	        													outFile_for_author_id_and_word_id_with_frequency,
			   	        													  -1, -1
			   									            				  , " load  auth id & doc id"
			   									            				  , false //isPrintSOP
			   									            				  );

			   	     writerDebug.append("mapIS_each_authorID_docID_CSV.size:"+mapIS_eachLINE_docID_WordID_Frequency.size()+" of file->"+outFile_for_author_id_and_word_id_with_frequency+"\n");
			   	     writerDebug.flush();
			   	     int lineNo=0; int doc_ID=-1;
			   	     System.out.println("Reading convert_each_author_each_doc_FREQ_to_each_auth_CUMM_FREQ() for file->inFile_1_for_each_doc_word_id_AND_FREQ"
			   	     					+ "	-> "+mapIS_eachLINE_docID_WordID_Frequency.size());
			   	     
			   	     writerDebug.append("map_StopWord_WordID:"+map_StopWord_WordID+"\n");
			   	     writerDebug.flush();
			   	     
			   	   	// Convert to map inFile_1_for_each_doc_word_id_AND_tf_idf
			   	     for(int curr_lineNo:mapIS_eachLINE_docID_WordID_Frequency.keySet()){
			 			lineNo++; 
			 			
			 			String curr_line=mapIS_eachLINE_docID_WordID_Frequency.get(curr_lineNo);
							//System.out.println("l:"+lineNumber++);
							if(curr_line.length()>=1){
								String conc_newLine="";
								///debug
								if(lineNo<20)
									System.out.println("lineNo:"+lineNo+"  line:"+line);
								
								String [] arr_tokens=curr_line.split("!!!"); //first token has doc_ID
								
								if(arr_tokens.length<2){
									System.out.println("arr_tokens:"+arr_tokens.length+" line:"+line);
									writerDebug.append(" not 2 token error: "+doc_ID+"\n");
									writerDebug.flush();
									continue;
								}
								doc_ID=Integer.valueOf(arr_tokens[0]);
								String [] arr_wordID_delimitr_TFIDF= arr_tokens[1].split(" ");
								int cnt=0;
								
								//
								while(cnt<arr_wordID_delimitr_TFIDF.length){
									
									String curr_wordID_delimitr_TFIDF =arr_wordID_delimitr_TFIDF[cnt]; //it has "233:1"
									String[] name_value=curr_wordID_delimitr_TFIDF.split(":");
									name_value[0]=name_value[0].trim();
									name_value[1]=name_value[1].trim();
									
									//if word ID corresponds to stop word, then skip
									if(  map_StopWord_WordID.containsValue( Integer.valueOf(name_value[0])  ) ){
										writerDebug.append(" remove stop word:"+name_value[0]);
										writerDebug.flush();
										cnt++;
										continue;
									}
									
									if(name_value.length<2){
										writerDebug.append(" not 2 token word:freq error: "+curr_wordID_delimitr_TFIDF+"\n");
										writerDebug.flush();
										continue;
									}
									////
									if(conc_newLine.length()==0){
										conc_newLine=curr_wordID_delimitr_TFIDF;
									}
									else{
										conc_newLine=conc_newLine+" "+curr_wordID_delimitr_TFIDF;
									}
									
									cnt++;
								} //while(cnt<arr_wordID_delimitr_TFIDF.length){
							
								
								writer_output.append(  doc_ID +"!!!"+  conc_newLine +"\n");
								writer_output.flush();
								
							} //if(curr_line.length()>=1){
							else{
								System.out.println("lineno:"+lineNo+" <1 so slip "  );
							}
							
							
							  
						} //while line-read
			
			
		} catch (Exception e) {
			e.printStackTrace();
			// TODO: handle exception
		}
		
		
	}
	
	//generate_NEW_IDs_auth_id_doc_id_queryTopic
	public static void generate_NEW_IDs_for_auth_id_AND_doc_id_for_each_queryTopic(
																  String baseFolder,
																  String inFile_curr_query_topic,
																  String outFile_authID_old_new,
																  String outFile_docID_old_new
																  ){
		BufferedReader reader=null; String line="";
		FileWriter writer_authID_old_new=null;
		FileWriter writer_docID_old_new=null;
		FileWriter writer_authID_NEW_docID_NEW=null;
		FileWriter writer_authID_NEW_docID_OLD=null;
		System.out.println("generate_NEW_ID->inFile_curr_query_topic:"+inFile_curr_query_topic);
			try{
				reader=new BufferedReader(new FileReader(new File(inFile_curr_query_topic)));
				writer_authID_old_new=new FileWriter(new File(outFile_authID_old_new));
				writer_docID_old_new=new FileWriter(new File(outFile_docID_old_new));
				writer_authID_NEW_docID_NEW=new FileWriter(new File(inFile_curr_query_topic+"_authID_NEW_docID_NEW.txt"));
				writer_authID_NEW_docID_OLD=new FileWriter(new File(inFile_curr_query_topic+"_authID_NEW_docID_OLD.txt"));
				TreeMap<Integer, Integer> mapAuth_ID_old_new=new TreeMap<Integer, Integer>();
				TreeMap<Integer, Integer> mapDocID_ID_old_new=new TreeMap<Integer, Integer>();
				// reading each line
				while((line=reader.readLine())!=null){
					   // current  queries
						String [] s=line.split("!!!");
						//
						if(!mapAuth_ID_old_new.containsKey( new Integer(s[0]) )){
							int sz=mapAuth_ID_old_new.size()+1;
							mapAuth_ID_old_new.put( new Integer(s[0]), sz);
							
						}
						if(!mapDocID_ID_old_new.containsKey( new Integer(s[1]) )){
							int sz=mapDocID_ID_old_new.size()+1;
							mapDocID_ID_old_new.put(new Integer(s[1]), sz);
						}
				}
				
				for(Integer i:mapAuth_ID_old_new.keySet()){
					writer_authID_old_new.append(i+"!!!"+mapAuth_ID_old_new.get(i)+"\n");
					writer_authID_old_new.flush();
				}
				//
				for(Integer i:mapDocID_ID_old_new.keySet()){
					writer_docID_old_new.append(i+"!!!"+mapDocID_ID_old_new.get(i)+"\n");
					writer_docID_old_new.flush();
				}
				
				reader=new BufferedReader(new FileReader(new File(inFile_curr_query_topic)));
				//reading again 
				while((line=reader.readLine())!=null){
					   // current  queries
						String [] s=line.split("!!!");
						
						
						writer_authID_NEW_docID_NEW.append(
									mapAuth_ID_old_new.get(new Integer(s[0]))+"!!!"+	
									mapDocID_ID_old_new.get(new Integer(s[1]))+"!!!"+"\n"
									);
						writer_authID_NEW_docID_NEW.flush();
						//
						writer_authID_NEW_docID_OLD.append(
								mapAuth_ID_old_new.get(new Integer(s[0]))+"!!!"+	
														s[1]+"!!!"+"\n"
								);
						writer_authID_NEW_docID_OLD.flush();
						
						
						
				}
				
				
			}
			catch(Exception e){
				e.printStackTrace();
			}
	}
	
	// "split_auth_id_doc_id_queryTopic" into each file with separate query topic 
	public static void split_auth_id_doc_id_queryTopic( String 					baseFolder,
														String 					outFile_auth_id_doc_id_queryTopicRelated_applying_cross_multiAuthIDs,
														TreeMap<Integer,String> map_10_Query_Topics
//														String 					outputFile_cross_multiAuthID
														){
		BufferedReader reader=null; String line="";
		try{
			FileWriter writerDebug=new FileWriter(new File(baseFolder+"debug_split_qtopic.txt"));
			System.out.println("inside split_auth_id_doc_id_queryTopic: given input map.size:"+map_10_Query_Topics.size());
			System.out.println("input file  :"+outFile_auth_id_doc_id_queryTopicRelated_applying_cross_multiAuthIDs);
			
			// below not required as "outFile_auth_id_doc_id_queryTopicRelated" corrected with applying multi AuthIDs
//		    TreeMap<Integer,String> map_seq_lines_from_cross_multi_authID=
//		            readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
//		            readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line
//																                    (
//																                    	 outputFile_cross_multiAuthID,
//																                         -1,
//																                         -1,  
//																                         " load11 ",    //debug label
//																                         false //isSOPprint
//																                    );
//		    TreeMap<Integer,Integer> map_authID_MINauthID =new TreeMap<Integer, Integer>();
//		    int count_multiauthid=0;
//		    //
//		    for(int seq:map_seq_lines_from_cross_multi_authID.keySet()){
//		    	if(seq==1) continue; //header
//		    	String []s = map_seq_lines_from_cross_multi_authID.get(seq).split("!!!");
//		    	String auth_name=s[0];
//		    	String all_authorids=s[4];
//		    	String min_authorid=s[5];
//		    	if(!min_authorid.equals(all_authorids)){
//		    		count_multiauthid++;
////		    		String new_all_authorids= all_authorids.replace(",", "#");
//		    		String [] s2=  all_authorids.split(",");
//		    		int cnt=0;
//		    		// 
//		    		while(cnt< s2.length){
//		    			if(!s2[cnt].equalsIgnoreCase(min_authorid  )  ){
//		    				System.out.println( "auth_name:"+auth_name+
//		    									" s2.length:"+s2.length +" s2[cnt]:"+s2[cnt]+" min_authorid:"+min_authorid //+" new_all_authorids:"+new_all_authorids
//		    											+" all_authorids:"+all_authorids);
//		    				map_authID_MINauthID.put( Integer.valueOf(s2[cnt]), Integer.valueOf(min_authorid) );
//		    				
//		    				writerDebug.append( auth_name +"!!!"+Integer.valueOf(s2[cnt]) +"!!!"+ Integer.valueOf(min_authorid)+"\n");
//		    				writerDebug.flush();
//		    			}
//		    			cnt++;
//		    		}
//		    	}
//		    }
//			
//		    System.out.println("map_authID_MINauthID:"+map_authID_MINauthID);
//		    System.out.println("count_multiauthid:"+count_multiauthid);
//			
//		    writerDebug.append("\n map_authID_MINauthID:"+map_authID_MINauthID);
//		    writerDebug.flush();
		    
		    //step : make new file out of "outFile_auth_id_doc_id_queryTopicRelated" to replace with  minAuthID from "map_authID_MINauthID"
		    
		    
			// comment/uncomment
			for(int numb:map_10_Query_Topics.keySet()){
				String curr_query_topic=map_10_Query_Topics.get(numb);
				/// 
				String out_=baseFolder+curr_query_topic+"_auth_id_doc_id_queryTopicRelated.txt";
				reader=new BufferedReader(new FileReader(new File(outFile_auth_id_doc_id_queryTopicRelated_applying_cross_multiAuthIDs)));
				FileWriter writer=new FileWriter(new File(out_ ));
				int curr_query_topic_count=0; int lineNo=0;
				// reading each line
				while((line=reader.readLine())!=null){
					// current  queries
					if(line.toLowerCase().indexOf(curr_query_topic)>=0){
						writer.append(line+"\n");
						writer.flush();
						curr_query_topic_count++;
					}
					lineNo++;
				}
				System.out.println("output :"+out_+" curr_query_topic_count:"+curr_query_topic_count+" lineNo:"+lineNo);
			} //for(int numb:map_10_Query_Topics.keySet()){
		}
		catch(Exception e){
			e.printStackTrace();
		}			
	}
	
	//is_given_author_a_dirtyAuthorName
	public static boolean is_given_author_a_dirtyAuthorName(
															String curr_authName,
															TreeMap<Integer, String> map_eachLine_DIRTY_authName,
															String currLine
															){
		try {
			
			for( int seq:map_eachLine_DIRTY_authName.keySet() ){
				
					String curr_dirty_authName=map_eachLine_DIRTY_authName.get(seq) ;
					
					//
					if(curr_dirty_authName.toLowerCase().indexOf(".line")>=0){
						if(currLine.indexOf(curr_dirty_authName.replace(".line", "")) >=0  )
							return true;
					}
					
					// 
					if(map_eachLine_DIRTY_authName.get(seq).indexOf(".equal")>=0 ){
						// 
						if(curr_authName.toLowerCase().equals( curr_dirty_authName.replace(".equal", "") ) ){
							return true;
						}
					}
					
					if(curr_authName.toLowerCase().indexOf( curr_dirty_authName ) >= 0 ){
						return true;
					}
					
					String curr_authName_noSpace=curr_authName.replace("from: ", "").replace("from:", "").replace(" ", "").toLowerCase();
					 // hard-coded 
					 if(curr_authName_noSpace.equals("theassociatepress") || curr_authName_noSpace.equals("associatepress")
					   || curr_authName_noSpace.equals("industrials")|| curr_authName_noSpace.equals("ifpri")
					   || curr_authName_noSpace.equals("home|armenianow.com")|| curr_authName_noSpace.equals("home")
					   || curr_authName_noSpace.equals("germany")|| curr_authName_noSpace.equals("boston")
					   || curr_authName_noSpace.equals("national")|| curr_authName_noSpace.equals("nation")
					   || curr_authName_noSpace.equals("bhutan observer")|| curr_authName_noSpace.equals("bihar")
					   || curr_authName_noSpace.equals("moderator")|| curr_authName_noSpace.equals("markets")
					   || curr_authName_noSpace.equals("more")|| curr_authName_noSpace.equals("whonews")
					   || curr_authName_noSpace.equals("tillbelowastridartnerrosemariesiebertstefansieber")
					   || curr_authName_noSpace.equals("team")|| curr_authName_noSpace.equals("teamfix")
					   || curr_authName_noSpace.equals("webmaster")|| curr_authName_noSpace.equals("nation")
					   || curr_authName_noSpace.equals("arutzshevanews")|| curr_authName_noSpace.equals("dev")
					   || curr_authName_noSpace.equals("guestblogger")|| curr_authName_noSpace.equals("author")
					   || curr_authName_noSpace.equals("burundinews")|| curr_authName_noSpace.equals("author")
					   || curr_authName_noSpace.equals("ndtvnewstamilnadu")|| curr_authName_noSpace.equals("ndtvnewskerala")
					   || curr_authName_noSpace.equals("rappler.com")|| curr_authName_noSpace.equals("manipurepao.nethotspot")
					   || curr_authName_noSpace.equals("andhrapradeshnews")|| curr_authName_noSpace.equals("ndtvnewskerala")
					   || curr_authName_noSpace.equals("diet")|| curr_authName_noSpace.equals("anewday")
					   || curr_authName_noSpace.equals("satffreporter")|| curr_authName_noSpace.equals("patna")|| curr_authName_noSpace.equals("sacw.net")
					   || curr_authName_noSpace.equals("tribunenewsservice") || curr_authName_noSpace.equals("turkey")
					   || curr_authName_noSpace.equals("tvguide") || curr_authName_noSpace.equals("turkey")
					   || curr_authName_noSpace.equals("buenosairesherald.com") || curr_authName_noSpace.equals("turkey")
					   || curr_authName_noSpace.indexOf(".net ")>=0
					   
							 ){
						  return true;
					 }
					
					
			}
			
		} catch (Exception e) {
			e.printStackTrace();
			// TODO: handle exception
		}
		return false;
	}
    
	// remove_DIRTY_authName_from_3Files
	public static void remove_DIRTY_authName_from_3Files(
															String  inFile_4_having_DIRTY_author_name,  // DIRTY AUTHOR NAMES // INPUT
															String  outFile_for_author_name_AND_author_id,// INPUT
															String  outFile_query_auth_id_doc_id_queryTopicRelated,// INPUT
															String  outFile_for_author_id_and_doc_ids, // INPUT
															String 	outFile_for_author_name_AND_author_id_NO_dirtyAuthNames, //OUT
															String 	outFile_for_author_name_AND_author_id_YES_dirtyAuthNames,//OUT
															String 	outFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames,//OUT
															String 	outFile_auth_id_doc_id_queryTopicRelated_YES_dirtyAuthNames,//OUT
															String  outFile_for_author_id_and_doc_ids_NO_dirtyAuthNames,//OUT
															String  outFile_for_author_id_and_doc_ids_YES_dirtyAuthNames//OUT
														){
		
		TreeMap<Integer, Integer> map_dirtyAuthID_as_KEY=new TreeMap<Integer, Integer>();
		TreeMap<Integer, String> map_eachLine_DIRTY_authName =new TreeMap<Integer, String>();
		try {
			
			FileWriter writer_outFile_for_author_name_AND_author_id_NO_dirtyAuthNames
							=new FileWriter(new File(outFile_for_author_name_AND_author_id_NO_dirtyAuthNames));
			FileWriter writer_outFile_for_author_name_AND_author_id_YES_dirtyAuthNames
							=new FileWriter(new File(outFile_for_author_name_AND_author_id_YES_dirtyAuthNames));
			
			FileWriter writer_outFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames
								=new FileWriter(new File(outFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames));
			FileWriter writer_outFile_auth_id_doc_id_queryTopicRelated_YES_dirtyAuthNames
								=new FileWriter(new File(outFile_auth_id_doc_id_queryTopicRelated_YES_dirtyAuthNames));
			
			FileWriter writer_outFile_for_author_id_and_doc_ids_NO_dirtyAuthNames
							=new FileWriter(new File(outFile_for_author_id_and_doc_ids_NO_dirtyAuthNames));
			FileWriter writer_outFile_for_author_id_and_doc_ids_YES_dirtyAuthNames
							=new FileWriter(new File(outFile_for_author_id_and_doc_ids_YES_dirtyAuthNames));
			
			// 
			TreeMap<Integer, String> map_eachLine_DIRTY_authName_INFILE =
			readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
			readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
																			inFile_4_having_DIRTY_author_name,
																			-1, 
																			-1, 
																			 "load bodytext",//debug_label,
																			 false // isPrintSOP
																			);
			
			//first column has author name
			for(int seq10:map_eachLine_DIRTY_authName_INFILE.keySet()){
				String [] s=map_eachLine_DIRTY_authName_INFILE.get(seq10).split("!!!");
				map_eachLine_DIRTY_authName.put(seq10, s[0] );
			}
			System.out.println("map_eachLine_DIRTY_authName_INFILE.size:"+map_eachLine_DIRTY_authName_INFILE.size());
			System.out.println("map_eachLine_DIRTY_authName.size:"+map_eachLine_DIRTY_authName.size());
			
			TreeMap<Integer, String> map_authName_authID =
					readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
					readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
																					 outFile_for_author_name_AND_author_id,
																					-1,
																					-1,
																					 "load bodytext",//debug_label,
																					 false // isPrintSOP
																					);
			
			TreeMap<Integer, String> map_authID_docID_query =
					readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
					readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
																					outFile_query_auth_id_doc_id_queryTopicRelated,
																					-1, 
																					-1, 
																					 "load bodytext",//debug_label,
																					 false // isPrintSOP
																					);
			TreeMap<Integer, String> map_authID_docID =
					readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
					readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
																					 outFile_for_author_id_and_doc_ids,
																					-1, 
																					-1, 
																					 "load bodytext",//debug_label,
																					 false // isPrintSOP
																					);
			
			String curr_sourceChannel="";
			// authName!!!authID!!!sourceChannelFile	<----first file
			 for(Integer seq:map_authName_authID.keySet()){
				 String curr_line_authName_authID=map_authName_authID.get(seq);
				 if(seq<=2) //DEBUG
				 	System.out.println("curr_line_authName_authID:"+curr_line_authName_authID);
				 String [] s=curr_line_authName_authID.split("!!!"); 
				 int curr_authID=new Integer( s[1]);
				 String curr_authName=s[0];
				 
				 if(s.length<3){ // curr_sourceChannel may be EMPTY or null
					 continue;
				 }
				 curr_sourceChannel=s[2];
				 
				 if(curr_sourceChannel.equalsIgnoreCase("null")) continue;
				 
				 boolean is_currAuthName_DIRTY=is_given_author_a_dirtyAuthorName(	
						 															curr_authName, //curr_authName
						 															map_eachLine_DIRTY_authName,
						 															"" // currLine
						 														);
				
				 
				 
				 // 
				 if(is_currAuthName_DIRTY || curr_authName.length() <=3 || curr_sourceChannel.equalsIgnoreCase("null") 
						 ){ //short author name can be dirty
					 
					 map_dirtyAuthID_as_KEY.put(curr_authID, -1);
					 
					 writer_outFile_for_author_name_AND_author_id_YES_dirtyAuthNames.append(curr_line_authName_authID+"\n");
					 writer_outFile_for_author_name_AND_author_id_YES_dirtyAuthNames.flush();
				 }
				 else{
					 writer_outFile_for_author_name_AND_author_id_NO_dirtyAuthNames.append(curr_line_authName_authID+"\n");
					 writer_outFile_for_author_name_AND_author_id_NO_dirtyAuthNames.flush();
				 }
				 
			 }
			 
			 ///// second file
			 for(int seq2:map_authID_docID_query.keySet()){
				 String currLine_authID_docID_query=map_authID_docID_query.get(seq2);
				 String []s=currLine_authID_docID_query.split("!!!");
				 int curr_AuthID=Integer.valueOf(s[0]) ; String curr_docID=s[1]; String curr_query=s[2];
				 
				 if(map_dirtyAuthID_as_KEY.containsKey(curr_AuthID)){
					 writer_outFile_auth_id_doc_id_queryTopicRelated_YES_dirtyAuthNames.append(currLine_authID_docID_query +"\n" );
					 writer_outFile_auth_id_doc_id_queryTopicRelated_YES_dirtyAuthNames.flush();
				 }
				 else{
					 writer_outFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames.append(currLine_authID_docID_query +"\n" );
					 writer_outFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames.flush();
				 }
			 }
			 System.out.println("map_authID_docID_query.size:"+map_authID_docID_query.size());
			 System.out.println("map_authID_docID.size:"+map_authID_docID.size());
			 System.out.println("map_dirtyAuthID_as_KEY.size:"+map_dirtyAuthID_as_KEY.size());
			 System.out.println("map_authName_authID.size:"+map_authName_authID.size());
			  //THIRD FILE 
			  for(int seq3:map_authID_docID.keySet()){
				  	String currLine_authID_docIDCSV =map_authID_docID.get(seq3);
				  	String []s=currLine_authID_docIDCSV.split("!!!");
				  	int curr_authID= new Integer(s[1]);
				  	String docID_csv=s[2];
				  	 
				  	if(map_dirtyAuthID_as_KEY.containsKey(curr_authID)){
				  		writer_outFile_for_author_id_and_doc_ids_YES_dirtyAuthNames.append(currLine_authID_docIDCSV+"\n");
				  		writer_outFile_for_author_id_and_doc_ids_YES_dirtyAuthNames.flush();
				  	}
				  	else{
				  		
				  		writer_outFile_for_author_id_and_doc_ids_NO_dirtyAuthNames.append(currLine_authID_docIDCSV+"\n");
				  		writer_outFile_for_author_id_and_doc_ids_NO_dirtyAuthNames.flush();
				  		
				  	}
				  
			  }
			
			
		} catch (Exception e) {
			e.printStackTrace();
			// TODO: handle exception
		}
	}
	
    // each document considered to have a bodyText
    // (1) output vocabulary word with unique ID
    // (2) output matrix of author_id and word_id:frequency for each document  => outFile_for_author_id_and_word_id_with_frequency
    // (3) output matrix of author_id and doc_id_1,doc_id_2,..,doc_id_N  => outFile_for_author_id_and_doc_ids_for_that_author
	// (4) TF_IDF feature vector of each document (each line) from "inFile_2_containing_bodyText_only"
    // It returns unique words (vocabulary) and unique ID
    public static TreeMap<String, Integer> p6_prepare_matrix_author_id_vs_docid_AND_unique_voc_word(
    												String  baseFolder,
                                                    String  inFile_1_all_tokens, //first column from this file will be doc_id OR line number of this file will become doc id
                                                    int     inFile_1_token_containing_body_text,
                                                    int     inFile_1_token_containing_author_names,
                                                    int     inFile_1_token_containing_source_URL,
                                                    String  inFile_1_pattern_to_find_keywords,
                                                    String  YES_CSV_Filter_for_inFile,
                                                    String  inFile_2_containing_bodyText_only,
                                                    String  inFile_3_having_unique_author_name,// each line has a author name
                                                    String  outFile_for_vocabulary,
                                                    String  outFile_for_author_name_AND_author_id,
                                                    String  outFile_for_authorNameNOSPACE_AND_authorNameVARIANT,
                                                    String  outFile_for_author_id_and_word_id_with_frequency,
                                                    String  outFile_for_author_id_and_doc_ids_for_that_author,
                                                    String  outFile_auth_id_doc_id_queryTopicRelated,
                                                    TreeMap<Integer,String> map_10_Query_Topics,
                                                    String  outFile_debugFile,
                                                    boolean is_load_vocabulary_from_past_run,
                                                    boolean isSOPprint
                                                    ){
        
            TreeMap<String, Integer> map_Word_UniqVocID;
                        map_Word_UniqVocID = new TreeMap<String, Integer>();
            TreeMap<Integer,String> map_UniqVocID_Word;
                        map_UniqVocID_Word = new TreeMap<Integer,String>();
                        
            TreeMap<String, Integer> map_AuthorName_AuthorID=new TreeMap<String,Integer>();
            TreeMap<String, String> map_AuthorName_sourceChannel=new TreeMap<String,String>();
            TreeMap<Integer, String> map_AuthorID_sourceChannel=new TreeMap<Integer,String>();
            TreeMap<String, Integer> map_AuthorNameNOSPACE_as_KEY_N_AuthID=new TreeMap<String,Integer>();
            TreeMap<String, String>  map_AuthorNameNOSPACE_AuthorNameVariant=new TreeMap<String, String>();
            
            TreeMap<Integer, String > map_docID_sourceURL=new TreeMap<Integer, String>();
            
            TreeMap<Integer, String> map_AuthorID_AuthorName=new TreeMap<Integer,String>();
            TreeMap<Integer,String>  map_authID_DocIDcsv=new TreeMap<Integer,String>();
            TreeMap<String,Integer>  map_authID_DocIDcsv_already=new TreeMap<String,Integer>();
            TreeMap<Integer,String>  map_Author_File=new TreeMap<Integer,String>();
            FileWriter writer_unique_voc_id=null;
            TreeMap<Integer, String > map_AuthorFileSeq_SourceChannel=new TreeMap<Integer, String>();
            FileWriter writer_debugFile=null;
        try{
        	if(is_load_vocabulary_from_past_run==false){
        		writer_unique_voc_id=new FileWriter(new File(outFile_for_vocabulary));
        	}
            FileWriter writer_docID_wordID_frequency=new FileWriter(new File(outFile_for_author_id_and_word_id_with_frequency));
            FileWriter writer_authName_AND_auth_ID=new FileWriter(new File(outFile_for_author_name_AND_author_id));
            FileWriter writer_authorNameNOSPACE_AND_authorNameVARIANT=new FileWriter(new File(outFile_for_authorNameNOSPACE_AND_authorNameVARIANT));
            
            FileWriter writer_authID_docID=new FileWriter(new File(outFile_for_author_id_and_doc_ids_for_that_author));
            FileWriter writer_auth_id_doc_id_queryTopicRelated=new FileWriter(new File(outFile_auth_id_doc_id_queryTopicRelated));
            writer_debugFile=new FileWriter(new File(outFile_debugFile));
            //has FILTERED document for queries
            TreeMap<Integer,String> mapInFile_1_for_document=new TreeMap<Integer, String>();
            System.out.println("LOADING INPUT FILE USING readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line");
            //get each document (a token from each line) from this file.
            //has UNFILTERED document for queries
            TreeMap<Integer,String> mapInFile_1_for_document_ORIG=
            readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
            readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line
														                    (
														                    	 inFile_1_all_tokens,
														                         -1,
														                         -1,  
														                         " load3 ",    //debug label
														                         false //isSOPprint
														                    );
            
            System.out.println("STEP: FILTERING OUT 10 QUERY TOPICS");
            // filter out 10 query topics
            for(int doc_id_:mapInFile_1_for_document_ORIG.keySet()){
            	String currLine=mapInFile_1_for_document_ORIG.get(doc_id_).toLowerCase();
                String []s=currLine.split("!!!");
                String curr_body_text=s[inFile_1_token_containing_body_text-1]
                                    .replace("bt:", "").replace(",", " ").replace(";", " ")
                                    .replace(":", " ").replace("”", " ").replace("“", " ")
                                    .replace("\"", "").replace(".", " ")
                                    .replace("\'", " ").replace("(", " ").replace(")", " ")
                                    .replace("’", " ").replace("?", " ").replace("‘", " ")
                                    .replace("  ", " ");
                curr_body_text=removeUnicodeChar.removeUnicodeChar(curr_body_text);
                
                String curr_Doc_sourceURL=s[inFile_1_token_containing_source_URL -1];
                
                map_docID_sourceURL.put(doc_id_, curr_Doc_sourceURL );
                
                boolean isCurrLine_has_any_one_query=false;
                //check_isCurrLine_has_any_one_query
                isCurrLine_has_any_one_query=cleaning_and_convert.check_isCurrLine_has_any_one_query( curr_body_text, map_10_Query_Topics );
            	
                // filter out only query topics (BELOW MAY NOT BE NEEDED as we already filter out )
                if(isCurrLine_has_any_one_query==false){
                	continue;
                }
                else{
                	mapInFile_1_for_document.put(doc_id_, mapInFile_1_for_document_ORIG.get(doc_id_));
                }
            	
            } //for(int doc_id_:mapInFile_1_for_document_ORIG.keySet()){
            
            System.out.println("LOADED mapInFile_1_for_document:"+mapInFile_1_for_document.size());
            
            System.out.println("LOADING FOR FILE->"+inFile_3_having_unique_author_name); 
            // get unique author name from input file
            TreeMap<Integer,String> map_Author_File_old=
            readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
            readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line
												                    (
												                         inFile_3_having_unique_author_name,
												                         -1,
												                         -1,  
												                         " load2 ",    //debug label
												                         false //isSOPprint
												                    );
            
            writer_debugFile.append("map_Author_File_old LOADED from file->"+inFile_3_having_unique_author_name+"\n");
            writer_debugFile.flush();
            String curr_SourceChannel="";
            // add this missing "auth:from:" PREFIX
            //eachline->AuthorName!!!AuthorNameOccuranceFreq_in_doc!!!SourceFileName!!!DomainNames
            for(int _seq:map_Author_File_old.keySet()){
            	if(_seq==1) continue; //HEADER
            	String []s=map_Author_File_old.get(_seq).split("!!!");
                String curr_author_name_atom=s[0];
                	   curr_SourceChannel=s[2];
                
                curr_author_name_atom=removeUnicodeChar.removeUnicodeChar(curr_author_name_atom);
                
                curr_author_name_atom=curr_author_name_atom.replace("auth:from: :from:", "auth:from: ")
			    	 					.replace("auth:from:  ", "auth:from: ").replace("auth:from::from:", "auth:from:")
			    	 					.replace("  ", " ");
                
                    if(curr_author_name_atom.indexOf("auth:from:")==-1 )
                        curr_author_name_atom="auth:from: "+curr_author_name_atom;
                    else if(curr_author_name_atom.indexOf("auth:from:")==-1  && 
                            curr_author_name_atom.indexOf("from:")>=0
                            ){
                        curr_author_name_atom="auth:"+curr_author_name_atom;
                    }
                    else{
                    	   curr_author_name_atom=curr_author_name_atom;
                    }
                    
                    //remove all
                    curr_author_name_atom=curr_author_name_atom.replace("auth:from: ", "").replace("from: ", "").replace("from:", "")
                    						.replace("auth:", "");
                    
                    // removing "rnd:"
                    if(curr_author_name_atom.indexOf(" rnd:")>=0){
                    	curr_author_name_atom= curr_author_name_atom.substring(0, curr_author_name_atom.indexOf(" rnd:"));
                    }
                    //
                    map_Author_File.put(_seq, curr_author_name_atom);
                    map_AuthorFileSeq_SourceChannel.put(_seq, curr_SourceChannel);
            } // END -- for(int _seq:map_Author_File_old.keySet()){
            
            writer_debugFile.append( "\n map_Author_File_old:"+map_Author_File_old);
            writer_debugFile.append( "\n map_Author_File:"+map_Author_File );
            writer_debugFile.flush();
            
            String body_text="";
            System.err.println("before global voc:");
            int total_mapInFile_1_for_document=mapInFile_1_for_document.size();
             int counter_=0;
            if(is_load_vocabulary_from_past_run==false){
	             // FRESH RUN ...
	             for(int i:mapInFile_1_for_document.keySet()){
	            	 counter_++;
	                System.out.println("mapInFile_1_for_document:counter_->"+counter_ +" out of total doc="+total_mapInFile_1_for_document);
	                
//	                if(i >1000) //debug break
//	                	break;
	                
	                try{
	                    body_text=mapInFile_1_for_document.get(i).split("!!!")[inFile_1_token_containing_body_text-1].replace("bt:", "");
	                }
	                catch(Exception e){
	                    System.out.println("error on body_text"+i);
	                    continue;
	                }
	                // unicode
	                body_text=removeUnicodeChar.removeUnicodeChar(body_text)
	                                    .replace("bt:", "").replace(",", " ").replace(";", " ")
	                                    .replace(":", " ").replace("”", " ").replace("“", " ")
	                                    .replace("\"", "").replace(".", " ")
	                                    .replace("\'", " ").replace("(", " ").replace(")", " ")
	                                    .replace("’", " ").replace("?", " ").replace("‘", " ")
	                                    .replace("   ", " ").replace("  ", " ");
	                
	                //replace blank space and special char with 
	                body_text = Replace_WhiteSpaceSpecialChar_with_3Hash(body_text);
	                
	                //String [] curr_line_split=body_text.split("\\s");
	                String [] curr_line_split=body_text.split("###");
	                
	                //System.out.println("body_text:"+body_text+" curr_line_split.len:"+curr_line_split.length);
	                
	                int c=0;
	                //each word of curr document
	                while(c<curr_line_split.length){
	                	//LOWER CASE
	                    String temp= removeUnicodeChar.removeUnicodeChar(
	                                 removeUnicodeChar.removeUnicodeChar(curr_line_split[c])).toLowerCase();
	                     
	                    		//compress.. will replace left out blank space..keeping only alpha numeric
	                    		temp=clean_retain_only_alpha_numeric_characters.clean_retain_only_alpha_numeric_characters(temp, false);
	                    
	                    int sz2=map_Word_UniqVocID.size()+1;
	                    //remove 
	                    
	                    //
	                    if(  stopwords.is_stopword(temp.toLowerCase())==false && !map_Word_UniqVocID.containsKey(temp.toLowerCase() ) ){
	                        map_Word_UniqVocID.put(temp.trim(), sz2);
	                        map_UniqVocID_Word.put(sz2, temp.trim());
	                        if(isSOPprint)
	                            System.out.println("new voc id:"+sz2+" =>"+temp);
	                    }
	                    c++;
	                }
	            } //END of for 
             
            }
            //LOAD it from past run
            else if(is_load_vocabulary_from_past_run==true){
            	TreeMap<String,String> map_UniqVocID_Word_LOADED=
            				LoadMultipleValueToMap2_AS_KEY_VALUE.LoadMultipleValueToMap2_AS_KEY_VALUE(outFile_for_vocabulary, "!!!",
            															"1,2",
            															"dummy"//append_suffix_at_each_line
																		, true   //is_have_only_alphanumeric
            															, false);
            	TreeMap<String,String> map_Word_UniqVocID_LOADED=
        				LoadMultipleValueToMap2_AS_KEY_VALUE.LoadMultipleValueToMap2_AS_KEY_VALUE_REVERSE(outFile_for_vocabulary, "!!!",
																							true,   //is_have_only_alphanumeric
        																					false);
            	//convert id to int
            	for(String vocID:map_UniqVocID_Word_LOADED.keySet()){
            		String word_=map_UniqVocID_Word_LOADED.get(vocID);
            		if(  stopwords.is_stopword(word_.toLowerCase())==false && !map_UniqVocID_Word.containsValue(word_.toLowerCase() ) ){
            			map_UniqVocID_Word.put(Integer.valueOf(vocID), map_UniqVocID_Word_LOADED.get(vocID));
            		}
            	}
            	////////////////////////////////////////
            	for(String word:map_Word_UniqVocID_LOADED.keySet()){
            		if(  stopwords.is_stopword(word.toLowerCase())==false && !map_Word_UniqVocID.containsKey(word.toLowerCase() ) ){
            			map_Word_UniqVocID.put(word, Integer.valueOf(map_Word_UniqVocID_LOADED.get(word)));
            		}
            	}
            	
            } 
            
            System.out.println("map_Word_UniqVocID:"+map_Word_UniqVocID.size());
            String curr_auth_name=""; String curr_SourceChannel2="";
            //create unique author ID for each author name. (Author Name will be normalized in this loop)
            //eachline->AuthorName!!!AuthorNameOccuranceFreq_in_doc!!!SourceFileName!!!DomainNames
             for(int i2:map_Author_File.keySet()){
                 System.out.println("create uniq:");
                 int sz=map_AuthorName_AuthorID.size()+1;
             	 // auth name and channel
                 curr_auth_name=map_Author_File.get(i2);
                 curr_SourceChannel2=map_AuthorFileSeq_SourceChannel.get(i2);
                 
                 curr_auth_name=removeUnicodeChar.removeUnicodeChar(curr_auth_name);
                 curr_auth_name=curr_auth_name.replace("auth:from: :from:", "auth:from: ")
						 	 					.replace("auth:from:  ", "auth:from: ").replace("auth:from::from:", "auth:from:")
						 	 					.replace("  ", " ").trim();
                  
                 writer_debugFile.append("\n create uniq auth=>"+curr_auth_name
                                            +"<=>"+sz);
                 writer_debugFile.flush();
                 // correct
                 if(   curr_auth_name.indexOf("auth:from: ")==-1
                      	&&curr_auth_name.indexOf("auth:from:")>=0){
                 	 curr_auth_name=curr_auth_name.replace("auth:from:", "auth:from: ");
                  }
                  else if(curr_auth_name.indexOf("auth:from:")==-1 ){
                 	 curr_auth_name=  curr_auth_name;
                      writer_debugFile.append("\n <map>auth:from:="+curr_auth_name);
                      writer_debugFile.flush();
                  }
                  else if(curr_auth_name.indexOf("auth:from:")==-1  && 
                 		 curr_auth_name.indexOf("from:")>=0
                          ){
                 	 curr_auth_name="auth:"+curr_auth_name;
                      writer_debugFile.append("\n <map>auth:from:(2)="+curr_auth_name);
                      writer_debugFile.flush();
                  }
                  else{
                	  curr_auth_name="auth:"+curr_auth_name; 
                  }
                 
                 //remove all
                 curr_auth_name=curr_auth_name.replace("auth:from: ", "").replace("from: ", "").replace("from:", "")
                 							  .replace("auth:", "");
                 
                 // ) missing, but ( present
                 if(curr_auth_name.indexOf("(")>=0 && curr_auth_name.indexOf(")") == -1  ){
                	 curr_auth_name=curr_auth_name +")";	 
                 }
                 //
                 curr_auth_name=curr_auth_name.replace("– reporting ", "").replace("", "").replace("reporting ", "")
                		 				      .replace("special correspondent", "").replace(" – ", "")
                		 				      .replace("— ", "").replace("– ", "");
                 
                 // NOT CONTAINS ADD 
                 if(  !map_AuthorNameNOSPACE_as_KEY_N_AuthID.containsKey( curr_auth_name.replace(" ", "") ) ){
                	//
                    map_AuthorName_AuthorID.put(curr_auth_name, sz);
                    map_AuthorID_AuthorName.put(sz, curr_auth_name);
                    // sourceChannel
                    map_AuthorName_sourceChannel.put(curr_auth_name, curr_SourceChannel2);
                    map_AuthorID_sourceChannel.put(  sz , curr_SourceChannel2);
                    // REMOVE space ( AUTHOR NORMALIZATION )
                    map_AuthorNameNOSPACE_as_KEY_N_AuthID.put( curr_auth_name.replace(" ", ""), sz);
                    
                 }
                 //CAPTURE the variants of Author Name
                 else if(!map_AuthorName_AuthorID.containsKey(curr_auth_name)
                      		&& map_AuthorNameNOSPACE_as_KEY_N_AuthID.containsKey( curr_auth_name.replace(" ", "") ) ) {
                	 // saving AUTHORname variant
                	 if(!map_AuthorNameNOSPACE_AuthorNameVariant.containsKey(curr_auth_name.replace(" ", ""))){
                		 map_AuthorNameNOSPACE_AuthorNameVariant.put( curr_auth_name.replace(" ", "") , curr_auth_name  );	 
                	 }
                	 else{
                		 String _old_variant=map_AuthorNameNOSPACE_AuthorNameVariant.get( curr_auth_name.replace(" ", "") );
                		 String new_all=_old_variant +"$$$"+curr_auth_name;
                		 //
                		 map_AuthorNameNOSPACE_AuthorNameVariant.put( curr_auth_name.replace(" ", "") , new_all);
                	 }
                 }
                 
             } //for(int i2:map_Author_File.keySet()){
             System.out.println("map_AuthorName_AuthorID:"+map_AuthorName_AuthorID.size()+" -"+map_AuthorName_AuthorID);
             
             writer_debugFile.append("----start------AUTH DICT NAME and ID \n");
             writer_debugFile.flush();
             ///DEBUG
             for(String name:map_AuthorName_AuthorID.keySet()){
            	 writer_debugFile.append("authname, id ->"+name+"!!!"+map_AuthorName_AuthorID.get(name)+"\n");
            	 writer_debugFile.flush();
             }
             writer_debugFile.append("----end------AUTH DICT NAME and ID \n");
             writer_debugFile.flush();

             //map_AuthorName_AuthorID
             writer_debugFile.append("\n map_AuthorName_AuthorID:"+map_AuthorName_AuthorID);
             writer_debugFile.flush();
             
             //Write authorNameNOSPACE and variant
             for(String Curr_authName: map_AuthorNameNOSPACE_AuthorNameVariant.keySet()){
            	 writer_authorNameNOSPACE_AND_authorNameVARIANT.append( Curr_authName+"!!!"+map_AuthorNameNOSPACE_AuthorNameVariant.get(Curr_authName)+" \n");
            	 writer_authorNameNOSPACE_AND_authorNameVARIANT.flush();
             }
             
             TreeMap<Integer,Integer> map_curr_doc_word_freq=new TreeMap<Integer,Integer>();
             String currLine="";
             int total_document =mapInFile_1_for_document.size();
             int doc_counter=0;
             // each line read (each line has a token =body_text of current document)
             // writing doc_id!!!word_id_1:freq word_id_2:freq
            for(int curr_docID:mapInFile_1_for_document.keySet()){
            	doc_counter++;
                writer_debugFile.append("\n =============");
                writer_debugFile.append("\n doc_ID:"+curr_docID);
                writer_debugFile.flush();
                
                //DUMMY DEBUG
                
                //
                System.out.println("each doc: "+doc_counter +" out of total doc="+total_document);
                                
                map_curr_doc_word_freq=new TreeMap();
                currLine=mapInFile_1_for_document.get(curr_docID).toLowerCase();
                //YES_filter
                String [] s = currLine.split("!!!");
                String curr_body_text=s[inFile_1_token_containing_body_text-1]
                                    .replace("bt:", "").replace(",", " ").replace(";", " ")
                                    .replace(":", " ").replace("”", " ").replace("“", " ")
                                    .replace("\"", "").replace(".", " ")
                                    .replace("\'", " ").replace("(", " ").replace(")", " ")
                                    .replace("’", " ").replace("?", " ").replace("‘", " ")
                                    .replace("  ", " ");
                curr_body_text=removeUnicodeChar.removeUnicodeChar(curr_body_text);
                
                String curr_author_names_CSV=s[inFile_1_token_containing_author_names-1];
                	   curr_author_names_CSV=curr_author_names_CSV.replace("from: ", "").trim();
                System.out.println("authors(CSV)->"+curr_author_names_CSV);
                
                
                TreeMap<String,String> map_currDoc_list_of_queryTopics_found=new TreeMap<String, String>();
                String conc_matrix_docID_wordID="";
                
                boolean isCurrLine_has_any_one_query=false;
                //check_isCurrLine_has_any_one_query
                isCurrLine_has_any_one_query=cleaning_and_convert.check_isCurrLine_has_any_one_query( curr_body_text, map_10_Query_Topics );
                
                // read 
                if( isCurrLine_has_any_one_query==false){
                	writer_debugFile.append("\n filtered out:doc_id:"+curr_docID);
                	writer_debugFile.flush();
                    continue;
                }
                else{
                	////////////////////
                	if(curr_body_text.toLowerCase().indexOf(map_10_Query_Topics.get(1))>=0)
                		map_currDoc_list_of_queryTopics_found.put(map_10_Query_Topics.get(1), "");
                	if(curr_body_text.toLowerCase().indexOf(map_10_Query_Topics.get(2))>=0)
                		map_currDoc_list_of_queryTopics_found.put(map_10_Query_Topics.get(2), "");
                	if(curr_body_text.toLowerCase().indexOf(map_10_Query_Topics.get(3))>=0)
                		map_currDoc_list_of_queryTopics_found.put(map_10_Query_Topics.get(3), "");
                	if(curr_body_text.toLowerCase().indexOf(map_10_Query_Topics.get(4))>=0)
                		map_currDoc_list_of_queryTopics_found.put(map_10_Query_Topics.get(4), "");
                	if(map_10_Query_Topics.containsKey(5)){
                		if(curr_body_text.toLowerCase().indexOf(map_10_Query_Topics.get(5))>=0)
                			map_currDoc_list_of_queryTopics_found.put(map_10_Query_Topics.get(5), "");
                	}
                	if(map_10_Query_Topics.containsKey(6)){
	                	if(curr_body_text.toLowerCase().indexOf(map_10_Query_Topics.get(6))>=0 )
	                		map_currDoc_list_of_queryTopics_found.put(map_10_Query_Topics.get(6), "");
                	}
                	if(map_10_Query_Topics.containsKey(7)){
	                	if(curr_body_text.toLowerCase().indexOf(map_10_Query_Topics.get(7))>=0)
	                		map_currDoc_list_of_queryTopics_found.put(map_10_Query_Topics.get(7), "");
                	}
                	if(map_10_Query_Topics.containsKey(8)){
	                	if(curr_body_text.toLowerCase().indexOf(map_10_Query_Topics.get(8))>=0)
	                		map_currDoc_list_of_queryTopics_found.put(map_10_Query_Topics.get(8), "");
                	}
                	
                	if(map_10_Query_Topics.containsKey(9)){
	                	if(curr_body_text.toLowerCase().indexOf(map_10_Query_Topics.get(9))>=0)
	                		map_currDoc_list_of_queryTopics_found.put(map_10_Query_Topics.get(9), "");
                	}
                	
                	if(map_10_Query_Topics.containsKey(10)){
	                	if(curr_body_text.toLowerCase().indexOf(map_10_Query_Topics.get(10))>=0)
	                		map_currDoc_list_of_queryTopics_found.put(map_10_Query_Topics.get(10), "");
                	}
                	
                }
                
                String [] arr_body_text=curr_body_text.toLowerCase().split("\\s");
                int c=0;
                int uniq_voc_id_for_word=-1;
                //each word of curr document
                while(c<arr_body_text.length){
                    //System.out.println("e.w.");
                    String temp= removeUnicodeChar.removeUnicodeChar(
                                 removeUnicodeChar.removeUnicodeChar(arr_body_text[c])).toLowerCase();
                    if(map_Word_UniqVocID.containsKey(temp)){
                        uniq_voc_id_for_word=map_Word_UniqVocID.get(temp);
                    }
                    else{
                        c++; continue;
                    }
                    if(map_curr_doc_word_freq.containsKey(uniq_voc_id_for_word )){
                        int sz3=map_curr_doc_word_freq.get(uniq_voc_id_for_word)+1;
                        map_curr_doc_word_freq.put(uniq_voc_id_for_word, sz3);
                    }
                    else{
                        map_curr_doc_word_freq.put(uniq_voc_id_for_word, 1);
                    }
                    c++;
                }
 
                //write the output to the matrix document
                for(int uniq_voc_id:map_curr_doc_word_freq.keySet()){
                	
                	///
                	if(map_UniqVocID_Word.containsKey(uniq_voc_id)){
                		
                		String curr_word=map_UniqVocID_Word.get(uniq_voc_id);
                		//STOPWORDS
                		if( stopwords.is_stopword( curr_word.toLowerCase() ) == false ){
                			//System.out.println("NO STOP WORDS: "+curr_word);
                			//System.out.println("each doc word");
    	                    if(conc_matrix_docID_wordID.length()==0){
    	                        conc_matrix_docID_wordID=uniq_voc_id+":"+map_curr_doc_word_freq.get(uniq_voc_id);
    	                    }
    	                    else{
    		                    conc_matrix_docID_wordID=conc_matrix_docID_wordID+" "+
    		                                            uniq_voc_id+":"+map_curr_doc_word_freq.get(uniq_voc_id);
    	                    }
                			
                		}
                		
                	}
                }
                //System.out.println("cm->"+conc_matrix_docID_wordID);
                //writing doc_id!!!word_id_1:freq word_id_2:freq 
                writer_docID_wordID_frequency.append("\n"+curr_docID+"!!!" +conc_matrix_docID_wordID); //problem to fix <-LENIN 14 MAY 
                writer_docID_wordID_frequency.flush();
                
                ///debug
                writer_debugFile.append("\n map_curr_doc_word_freq:"+map_curr_doc_word_freq
                						//+" conc_matrix_docID_wordID:"+conc_matrix_docID_wordID
                						+" conc_matrix_docID_wordID.len:"+conc_matrix_docID_wordID.length());
                writer_debugFile.flush();
                
                //split
                //String [] arr_author_name_atom=curr_author_names_CSV.split(",");
                int c10=0;
                int curr_auth_id=1;
                
                 TreeMap<String,String> mapAuthorNames=new TreeMap<String,String>();
                 String[] s_token=null;
                 String[] s_token_2=null;
                 //BOTH <and> <,> => matt apuzzo, mark mazzetti and michael s. schmidt
                 mapAuthorNames=splitCSVAuthors_to_atom.
                                splitCSVAuthors_to_atom(curr_author_names_CSV,"", isSOPprint);
                 
                
                //
                //while(c10<arr_author_name_atom.length){
                 for(String arr_author_name_atom2:mapAuthorNames.keySet()){
                	 if(isSOPprint)
                		 System.out.println("author atom");
                    //after atom split - "auth:from:" is missing for second, third... authors
                    String curr_author_name_atom=arr_author_name_atom2;
                    curr_author_name_atom=removeUnicodeChar.removeUnicodeChar(curr_author_name_atom);
                    curr_author_name_atom=curr_author_name_atom.replace("auth:from: :from:", "auth:from: ")
                    						.replace("auth:from:  ", "auth:from: ").replace("auth:from::from:", "auth:from:")
                    						.replace("  ", " ").trim();
                    
                    // replace adustment for auth:from:
                    if(   curr_author_name_atom.indexOf("auth:from: ")==-1
                    	&&curr_author_name_atom.indexOf("auth:from:")>=0){
                    	curr_author_name_atom=curr_author_name_atom.replace("auth:from:", 
                    														"auth:from: ");
                    }
                    else if(curr_author_name_atom.indexOf("auth:from:")==-1 ){
                        curr_author_name_atom= curr_author_name_atom;
                        writer_debugFile.append("\nadded:"+curr_author_name_atom);
                        writer_debugFile.flush();
                    }
                    else if(curr_author_name_atom.indexOf("auth:from:")==-1  && 
                            curr_author_name_atom.indexOf("from:")>=0
                            ){
                        curr_author_name_atom="auth:"+curr_author_name_atom;
                        writer_debugFile.append("\nadded auth:from:(2)="+curr_author_name_atom);
                        writer_debugFile.flush();
                    }
                    else{
                    	curr_author_name_atom="auth:"+curr_author_name_atom; 
                    }
                    
                    //remove all
                    curr_author_name_atom=curr_author_name_atom.replace("auth:from: ", "").replace("from: ", "").replace("from:", "")
                    							  				 .replace("auth:", "").trim();
                    //
                    curr_author_name_atom=curr_author_name_atom.replace("  ", " ").replace("  ", " ");
                    // removing "rnd:"
                    if(curr_author_name_atom.indexOf(" rnd:")>=0){
                    	curr_author_name_atom= curr_author_name_atom.substring(0, curr_author_name_atom.indexOf(" rnd:"));
                    }
                    //
                    String curr_author_name_atom_NOSPACE=curr_author_name_atom.replace(" ","");
                    //
                    if(curr_author_name_atom.indexOf("(")>=0 &&  curr_author_name_atom.indexOf(")")==-1){
                    	curr_author_name_atom=curr_author_name_atom+")";
                    }
                    // author ID present?
                    if(map_AuthorName_AuthorID.containsKey(curr_author_name_atom)
                    	|| map_AuthorNameNOSPACE_as_KEY_N_AuthID.containsKey(curr_author_name_atom_NOSPACE )
                    	|| map_AuthorName_AuthorID.containsKey( "auth:from: "+curr_author_name_atom )
                    						){
                    	
                    	writer_debugFile.append("\n NOW curr_author_name_atom:"+curr_author_name_atom.replace("  ", " "));
                    	writer_debugFile.flush();
                    	/// get Auth id
                    	if(map_AuthorNameNOSPACE_as_KEY_N_AuthID.containsKey(curr_author_name_atom_NOSPACE )){
                    		curr_auth_id=map_AuthorNameNOSPACE_as_KEY_N_AuthID.get(curr_author_name_atom_NOSPACE);
                    	}
                    	else if( map_AuthorName_AuthorID.containsKey(curr_author_name_atom ) ){
                    		curr_auth_id=map_AuthorName_AuthorID.get(curr_author_name_atom);	
                    	}
                    	else if(map_AuthorName_AuthorID.containsKey( "auth:from: "+curr_author_name_atom ))  {
                    		curr_auth_id=map_AuthorName_AuthorID.get("auth:from: "+curr_author_name_atom );
                    	}
                        
                        // <curr_authid!!!curr_docID!!!currQuery>
                        for(String currQuery:map_currDoc_list_of_queryTopics_found.keySet()){ 
	                        writer_auth_id_doc_id_queryTopicRelated.append(curr_auth_id+"!!!"+
	                        											   curr_docID+"!!!"+ // auth_id, doc_id , queryTopic
	                        											   currQuery
	                        											   +"\n"
	                        											  );
	                        writer_auth_id_doc_id_queryTopicRelated.flush();
                        }
                        
                    }
                    else{
                    	if(isSOPprint)
	                        System.out.println(" cant find author ID: "+curr_author_name_atom
	                        				+"=>"+curr_author_name_atom.replace("  ", " ")
	                                        +" lineNumber:"+curr_docID);
                    	
                    	int auth_id_NEW=map_AuthorName_AuthorID.size()+1;

                    	// INSERT NEW AUTH ID
                    	if(!map_AuthorID_AuthorName.containsKey(auth_id_NEW) &&
                    			curr_author_name_atom.indexOf(" rnd:" ) ==-1
                    			){
	                    	map_AuthorNameNOSPACE_as_KEY_N_AuthID.put(curr_author_name_atom_NOSPACE, auth_id_NEW);
	                    	map_AuthorName_AuthorID.put(curr_author_name_atom, auth_id_NEW);
	                    	map_AuthorID_AuthorName.put(auth_id_NEW, curr_author_name_atom);
	                    	
	                        // <curr_authid!!!curr_docID!!!currQuery>
	                        for(String currQuery:map_currDoc_list_of_queryTopics_found.keySet()){ 
		                        writer_auth_id_doc_id_queryTopicRelated.append(curr_auth_id+"!!!"+curr_docID+"!!!"+ // auth_id, doc_id , queryTopic
		                        											   currQuery
		                        											   +"\n"
		                        												);
		                        writer_auth_id_doc_id_queryTopicRelated.flush();
	                        }
                    	}
                    	else{
	                    	//
	                        writer_debugFile.append("\n cant find author ID: "+curr_author_name_atom
	                                                +" lineNumber:"+curr_docID);
	                        writer_debugFile.flush();
                    	}
                        
                        c10++; 
                        continue;
                    }
                    
                    // map_authID_DocIDcsv_already
                    if(!map_authID_DocIDcsv_already.containsKey(curr_auth_id+"!!!"+String.valueOf(curr_docID))){
                        //
                        if(!map_authID_DocIDcsv.containsKey(curr_auth_id)){
                            map_authID_DocIDcsv.put(curr_auth_id, String.valueOf(curr_docID));
                            writer_debugFile.append("\n 0. $$$ "+curr_auth_id+"<=>"+curr_docID);
                            writer_debugFile.flush();
                            map_authID_DocIDcsv_already.put(curr_auth_id+"!!!"+String.valueOf(curr_docID) //already recorded doc ID
                                                            , curr_docID); 
                        }
                        else{
                            String DocIDcsv=map_authID_DocIDcsv.get(curr_auth_id)+","+curr_docID;
                            writer_debugFile.append("\n 1. $$$ "+curr_auth_id+"<=>"+curr_docID);
                            writer_debugFile.flush();
                            
                            map_authID_DocIDcsv.put(curr_auth_id, DocIDcsv);
                            map_authID_DocIDcsv_already.put(curr_auth_id+"!!!"+String.valueOf(curr_docID) //already recorded doc ID
                                                            , curr_docID ); 
                        }
                    }
                    else{
                        writer_debugFile.append("\n 2. $$$ "+curr_auth_id
                                                +"<=>"+String.valueOf(curr_docID));
                        writer_debugFile.flush();
                    }
                    
                    System.out.println("map_authID_DocIDcsv:size:"
                                        +map_authID_DocIDcsv.size());
                    c10++;
                } //end WHILE ON arr_author_name_atom
                System.out.println("mapInFile_1_for_document:size:"+mapInFile_1_for_document.size());
            } //end for each document
            int size_v=map_Word_UniqVocID.size();
            int cnt=0;
            
            try{
                //write vocabulary output file
                while(cnt<size_v){
                    if(map_UniqVocID_Word.containsKey(cnt))
                        writer_unique_voc_id.append(cnt+"!!!"+map_UniqVocID_Word.get(cnt)+"\n");
                    writer_unique_voc_id.flush();
                    cnt++;
                }
            }
            catch(Exception e){
            	System.out.println("ERROR !! ");
            	writer_debugFile.append("\n error 1 : "+e.getMessage());
            	writer_debugFile.flush();
                e.printStackTrace();
            }
            int c2=1;
            //writing authID and doc_id_CSV
            //for(int h:map_authID_DocIDcsv.keySet()){
      
                //while(c2<map_AuthorName_AuthorID.size()) {
                for(int h:map_authID_DocIDcsv.keySet()){
                    try{
                     
                        writer_authID_docID.append("!!!"+h+"!!!"+map_authID_DocIDcsv.get(h)+"\n");
                        writer_authID_docID.flush();
                    }
                    catch(Exception e){
                    	writer_debugFile.append("\n error 2 : "+e.getMessage());
                    	writer_debugFile.flush();
                        e.printStackTrace();
                    }
                }
                //write AUTH NAME and AUTH ID
                for(String cname:map_AuthorName_AuthorID.keySet()){
                    try{
                        writer_authName_AND_auth_ID.append(cname+"!!!"
                                                            +map_AuthorName_AuthorID.get(cname)+"!!!"
                                                            +map_AuthorID_sourceChannel.get(map_AuthorName_AuthorID.get(cname))
                                                            +"\n"
                                                            );
                        writer_authName_AND_auth_ID.flush();
                    }
                    catch(Exception e){
                    	writer_debugFile.append("\n error 3 : "+e.getMessage());
                    	writer_debugFile.flush();
                        e.printStackTrace();
                    }
                }
                
            	////  STEP :  CROSS Author Names that belong to same person/entity but has many Auth IDs
//            	cross_map_multi_authIDs_of_same_AuthName(
//            												baseFolder,
//            												outFile_for_author_name_AND_author_id //input file
//            											); 
                
            // calculate Tf-IDF from the bodyText File
//              tf_idf_AND_cosine_main.calc_tf_idf_for_one_File( inFile_2_containing_bodyText_only, 
//            		  										   baseFolder+"tf_idf_each_doc.txt");
             
            System.out.println("map_Word_UniqVocID.size:"+map_Word_UniqVocID.size());
            System.out.println("mapInFile_1_for_document:size:"+mapInFile_1_for_document.size());
            System.out.println("map_AuthorName_AuthorID.size:"+map_AuthorName_AuthorID.size());
            System.out.println("map_AuthorID_AuthorName:"+map_AuthorID_AuthorName);
            
            
            writer_debugFile.append("\nmap_AuthorName_sourceChannel:"+map_AuthorName_sourceChannel+"\n");
            writer_debugFile.append("\nmap_AuthorName_sourceChannel.size:"+map_AuthorName_sourceChannel.size()+"\n");
            writer_debugFile.append("\nmapInFile_1_for_document_ORIG.size:"+mapInFile_1_for_document_ORIG.size()+"\n");
            writer_debugFile.append("\nmapInFile_1_for_document:size:"+mapInFile_1_for_document.size()+"\n");
            writer_debugFile.flush();
            
        }
        catch(Exception e){
        	try{
	        	writer_debugFile.append("\n catch error: "+e.getMessage());
	        	writer_debugFile.flush();
        	}
        	catch(Exception e2){
        	}
            e.printStackTrace();
        }
        System.out.println("Success..");
        return map_Word_UniqVocID;
    }
    
    //generate_pageRANK_inputFile_each_auth_pair_COSINE
	private static void generate_pageRANK_INPUT_inputFile_each_auth_pair_COSINE(
																		  String baseFolder,
																		  String inFile_curr_q_topicWISE_authID_docID,
																		  String inFile_outFile_for_author_id_and_word_id_with_frequency,
																		  String out_topicWISE_authID_docID_CSV
																		  ){
		BufferedReader reader=null; String line="";
		TreeMap<Integer, String> map_authID_docID_CSV=new TreeMap<Integer, String>();
		try{
			File f_inFile_curr_q_topicWISE_authID_docID=new File(inFile_curr_q_topicWISE_authID_docID);
			
			reader=new BufferedReader(new FileReader(new File(inFile_curr_q_topicWISE_authID_docID)));
			//new File(inFile_outFile_for_author_id_and_word_id_with_frequency+"_docIDCSV.txt").createNewFile();
			FileWriter writer=new FileWriter(new File(out_topicWISE_authID_docID_CSV));
			// each line 
			while((line=reader.readLine())!=null){
				String [] s=line.split("!!!");
				if(!map_authID_docID_CSV.containsKey(new Integer(s[0]))){
					map_authID_docID_CSV.put(new Integer(s[0]),  s[1]); //only auth_ID
				}
				else{
					String tmp=map_authID_docID_CSV.get(new Integer(s[0])) +","+s[1];
					map_authID_docID_CSV.put(new Integer(s[0]), tmp);
				}
			}
			// 
			//   <auth_id!!!doc_ID_CSV>
			for(int j:map_authID_docID_CSV.keySet()){
				//first column empty
				writer.append("!!!"+j+"!!!"+map_authID_docID_CSV.get(j)+"\n");
				writer.flush();
			}
			//calculate cosine similarit
			writer.close();
		
			
			FileWriter writerDebug=new FileWriter(new File(baseFolder+"debug2.txt"),true);
			
			writerDebug.append("\n 1:baseFolder"+baseFolder);
			writerDebug.append("\n 2:"+baseFolder+"doc_id_word_id_AND_freq_d.txt");
			writerDebug.append("\n 3:"+inFile_outFile_for_author_id_and_word_id_with_frequency+"_docIDCSV.txt");
			writerDebug.append("\n 4:"+baseFolder+"pagerank-cosine-for-"+f_inFile_curr_q_topicWISE_authID_docID.getName());
			
			writerDebug.append("\n 3: file exist:"
									+"<-->"+new File(inFile_curr_q_topicWISE_authID_docID).exists());
			
			writerDebug.append("\n 5: out_topicWISE_authID_docID:"+out_topicWISE_authID_docID_CSV);
			writerDebug.flush();

			
		}
		catch(Exception e){
			e.printStackTrace();
		}
		
	}

    // generate_pageRANK_inputFile_with_new_AuthID
	private static void generate_pageRANK_inputFile_with_new_AuthID(
										String inFile_authID_docID_QTopic_old, 
										String inFile_outFile_authID_old_new, 
										String outFile_pageRank_inFile) {
		// TODO Auto-generated method stub
		BufferedReader reader=null;
		String line="";
		try{
			reader=new BufferedReader(new FileReader(new File(inFile_authID_docID_QTopic_old)));
			TreeMap<Integer, Integer> map_authID_docID_qTopic_old=new TreeMap<Integer, Integer>();
			TreeMap<Integer, Integer> map_authID_old_new_mapping=new TreeMap<Integer, Integer>();
			while((line=reader.readLine())!=null){
				   // current  queries
					String [] s=line.split("!!!");
					map_authID_docID_qTopic_old.put(new Integer(s[0]), new Integer(s[1]));
			}
			
			reader=new BufferedReader(new FileReader(new File(inFile_outFile_authID_old_new)));
			while((line=reader.readLine())!=null){
				   // current  queries
					String [] s=line.split("!!!");
					map_authID_old_new_mapping.put(new Integer(s[0]), new Integer(s[1]));
			}
			
		}
		catch(Exception e){
			e.printStackTrace();
		}
		
	}
	 
	 // input_authID_pair_out_unique_authID
    public static TreeMap<Integer, Integer> input_authID_pair_out_unique_authID_for_inFile_of_COSINE
                                                                                (
                                                                                String baseFolder,
                                                                                String inFile,
                                                                                String outFile_1_cosine,
                                                                                String outFile_2_pagerank_mapping
                                                                                ){
        BufferedReader reader=null;
        String line=""; FileWriter writer=null;
        TreeMap<Integer, Integer> mapAuthId_unique_old_new=new TreeMap<Integer, Integer>();
        int max_=-1; FileWriter writer2_pageRank=null;
        try{
            FileWriter writerDebug=new FileWriter(new File(baseFolder+"debug_authpair_toUNIQ.txt"));
            writer =new FileWriter(new File(outFile_1_cosine));
            writer2_pageRank= new FileWriter(new File(outFile_2_pagerank_mapping));
            
            writerDebug.append("\n 1: "+baseFolder+" \n 2:"+inFile+"\n 3:"+outFile_1_cosine);
            writerDebug.flush();
            
            reader=new BufferedReader(
                        new FileReader(new File(inFile) ));
            // each Line
            while((line=reader.readLine())!=null){
               
                String s[]=line.split(",");
                int sz=-1;
                if(!mapAuthId_unique_old_new.containsKey(new Integer(s[0]))){
                    sz=mapAuthId_unique_old_new.size()+1;
                    mapAuthId_unique_old_new.put(new Integer(s[0])  , sz );   
                }
               
                if(!mapAuthId_unique_old_new.containsKey(new Integer(s[1]))){
                    sz=mapAuthId_unique_old_new.size()+1;
                    mapAuthId_unique_old_new.put(new Integer(s[1])  , sz );
                }
               
            }// end while
           
            reader=new BufferedReader(new FileReader(new File(inFile) ));
            // each file
            while((line=reader.readLine())!=null){
                String s[]=line.split(",");
               
                writer.append(
                        mapAuthId_unique_old_new.get(new Integer(s[0]))
                        +","+
                        mapAuthId_unique_old_new.get(new Integer(s[1]))
                        +","+
                        s[2]
                        +"\n"
                        );
                writer.flush();
               
            }// end while
       
            // read each old auth_id and its new 
            for(int j:mapAuthId_unique_old_new.keySet()){
            	writer2_pageRank.append(j+"!!!"+mapAuthId_unique_old_new.get(j)+"\n");
            	writer2_pageRank.flush();
            }
            
            writerDebug.append("\nmapAuthId_unique_old_new.size:"+mapAuthId_unique_old_new.size());
            writerDebug.flush();
       
            //writerDebug.close();
        }
        catch(Exception e){
            e.printStackTrace();
        }
        return mapAuthId_unique_old_new;
    }
    
    /////cross_map_multi_authIDs_of_same_AuthName
    public static void cross_map_multi_authIDs_of_same_AuthName(
    															String baseFolder,
    															String outFile_for_author_name_AND_author_id, // input file
    															String outputFile_cross_multiAuthID,
    															String outputFile_cross_multiAuthID_SORTED,
    															String outputFile_cross_multiAuthID_SORTED_addMINauthID
    															) {
		// TODO Auto-generated method stub
    	try {
    		 
    		FileWriter writer=new FileWriter(new File( outputFile_cross_multiAuthID ));
    		FileWriter writer_SORTED=new FileWriter(new File(outputFile_cross_multiAuthID_SORTED ));
    		FileWriter writer_SORTED_addMINauthID=new FileWriter(new File(outputFile_cross_multiAuthID_SORTED_addMINauthID ));
    		FileWriter writerDebug=new FileWriter(new File( baseFolder+"debug_corss_multiAuthID.txt" ));
    		//
    		TreeMap<String,String> map_AuthName_AuthID_LOADED=
     				LoadMultipleValueToMap2_AS_KEY_VALUE.LoadMultipleValueToMap2_AS_KEY_VALUE(outFile_for_author_name_AND_author_id, 
     																							"!!!",
     																							"1,2",
     																							"", //append_suffix_at_each_line
     																							false,   //is_have_only_alphanumeric
     																							false);
    		
    		// readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
    		TreeMap<Integer, String > map_seq_line=
			    		readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
			    		readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(outFile_for_author_name_AND_author_id,
			    																					-1 , //startline, 
			    																					-1, //endline
			    																					"debug_label", 
			    																					false //isPrintSOP
			    																					);
    		TreeMap<String, String> map_authName_eachLine=new TreeMap<String,String>();
    		TreeMap<String, String> map_lineNo_eachLine=new TreeMap<String,String>();
    		// PUT authName and LINE 
    		for(Integer seq:map_seq_line.keySet()){
    			String eachLine=map_seq_line.get(seq);
    			String []s=eachLine.split("!!!");   			
    		}
    		
    		TreeMap<String, String> map_AuthName_concAuthIDs =new TreeMap<String, String>();

//    		System.out.println("map_AuthName_AuthID_LOADED:"+map_AuthName_AuthID_LOADED);
    		
    		// OUT
    		for(String out_AuthName:map_AuthName_AuthID_LOADED.keySet()){
    			
    			if(map_AuthName_AuthID_LOADED.get(out_AuthName) == null || 
    					map_AuthName_AuthID_LOADED.get(out_AuthName).equals("")  )
    				continue;
    			
    			int 	out_AuthID = Integer.valueOf(map_AuthName_AuthID_LOADED.get(out_AuthName));
    			boolean is_found_cross_matching=false;
    			// IN
    			for(String in_AuthName:map_AuthName_AuthID_LOADED.keySet()){
        			int in_AuthID = Integer.valueOf( map_AuthName_AuthID_LOADED.get(in_AuthName) );
        			// matching similar name with different auth ID
        			if( out_AuthID != in_AuthID && in_AuthName.indexOf( out_AuthName )>=0 ){
        				is_found_cross_matching=true;
        				//
        				if( !map_AuthName_concAuthIDs.containsKey( out_AuthName) ){
        					map_AuthName_concAuthIDs.put(out_AuthName, String.valueOf(in_AuthID) );	
        				}
        				else{
        					String conc_AuthIDS_ =  map_AuthName_concAuthIDs.get(out_AuthName);
        					conc_AuthIDS_=conc_AuthIDS_+"$$$"+ String.valueOf(in_AuthID);
        					map_AuthName_concAuthIDs.put(out_AuthName, conc_AuthIDS_);
        				}
        			}
    			}
    			
    			
    			//junk names as author names, then continue....dont write "writer.append"
				if(out_AuthName.indexOf("2015") >=0 ||out_AuthName.indexOf("syrian") >=0 ||out_AuthName.indexOf("mozilla") >=0
				  ||out_AuthName.indexOf("2016") >=0 ||out_AuthName.equalsIgnoreCase("(reporting)")||out_AuthName.indexOf(" from ") >=0
				  ||out_AuthName.indexOf(" dies") >=0||out_AuthName.indexOf(" government") >=0||out_AuthName.indexOf(" suggest") >=0
				  ||out_AuthName.indexOf(" know ") >=0||out_AuthName.indexOf(" killed")>=0||out_AuthName.equalsIgnoreCase("world")
				  ||out_AuthName.indexOf(" get ") >=0 ||out_AuthName.indexOf(" children ") >=0||out_AuthName.indexOf("pakistani") >=0
				  ||out_AuthName.indexOf("pic.twitter.com") >=0  ||out_AuthName.equalsIgnoreCase("syria")||out_AuthName.indexOf("000") >=0
				  ){
					continue;
				}
    			//
    			if(is_found_cross_matching){
    				 
    				String [] s=map_AuthName_concAuthIDs.get(out_AuthName).split("\\$\\$\\$");
//    				System.out.println("wriitng: "+out_AuthName+" "+s.length);
    				//     				
    				if(s.length<100){
	    				writer.append(out_AuthName+"!!!"+ out_AuthID+"!!!"+map_AuthName_concAuthIDs.get(out_AuthName)+"\n");
	    				writer.flush();
    				}
    				else{
    					writerDebug.append(">100->"+out_AuthName+"!!!"+ out_AuthID+"!!!"+map_AuthName_concAuthIDs.get(out_AuthName)+"\n");
    					writerDebug.flush();
    				}
    			}
    			else{
    				String [] s= String.valueOf(out_AuthID).split("\\$\\$\\$");
//    				System.out.println("wriitng2: "+out_AuthName+" "+s.length);
    				// 
    				if(s.length<100){
    					writer.append(out_AuthName+"!!!"+ out_AuthID+"!!!"+"\n");
    					writer.flush();
    				}
    				else{
    					writerDebug.append(">100->"+out_AuthName+"!!!"+ out_AuthID+"!!!"+out_AuthID+"\n");
    					writerDebug.flush();
    				}
    			}
    			
    		}
    		writer.close();
    		
    		// readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
    		TreeMap<Integer, String > map_seq2_line=
			    		readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
			    		readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(outputFile_cross_multiAuthID,
			    																					-1 , //startline, 
			    																					-1, //endline
			    																					"debug_label", 
			    																					false //isPrintSOP
			    																					);
    		 
    		TreeMap<Integer, String> map_lineNo2_eachLine=new TreeMap<Integer,String>();
    		TreeMap<Integer, String> map_lineNo2_authName=new TreeMap<Integer,String>();
    		TreeMap<Integer, String> map_lineNo2_authId=new TreeMap<Integer,String>();
    		TreeMap<Integer, String> map_lineNo2_cross_multi_authids=new TreeMap<Integer,String>();
    		 Map<Integer, Integer> map_lineNo2_size_OF_cross_multi_authids=new TreeMap<Integer,Integer>();
    		// PUT authName and LINE 
    		for(Integer seq:map_seq2_line.keySet()){
    			String eachLine=map_seq2_line.get(seq);
    			map_lineNo2_eachLine.put(seq, eachLine);
    			String []s=eachLine.split("!!!");
    			String authname=""; String authid=""; String cross_multi_authids="";
    			if(s.length>=3){
    				authname=s[0];
    				authid=s[1];
    				cross_multi_authids=s[2];
    				String [] s2=cross_multi_authids.split("\\$\\$\\$");
    				int _len=s2.length;
    				map_lineNo2_authName.put(seq, authname);
    				map_lineNo2_authId.put(seq, authid);
    				map_lineNo2_cross_multi_authids.put(seq, cross_multi_authids);
    				map_lineNo2_size_OF_cross_multi_authids.put( seq , _len);
    			}
    			else if(s.length==2){
    				authname=s[0];
    				authid=s[1];
    				map_lineNo2_authName.put(seq, authname);
    				map_lineNo2_authId.put(seq, authid);
    				map_lineNo2_size_OF_cross_multi_authids.put( seq , 0);
    			}
    		}
    		// SORTING
    		 Map<Integer, Integer> tmp_Sorted =sort_given_treemap.sortByValue_II_method4_ascending(map_lineNo2_size_OF_cross_multi_authids);
    		 
    		writer_SORTED.append("AuthName!!!AuthID!!!cross-MultiAuthIDmatched!!!distinctCount-cross-MultiAuthIDmatched\n");
    		
    		//writer sorted by total unique authIDs
    		for(Integer _seq:tmp_Sorted.keySet()){
    			
    			writer_SORTED.append( map_lineNo2_eachLine.get( Integer.valueOf( _seq) )+"!!!"+ tmp_Sorted.get(_seq) +"!!!\n");
    			writer_SORTED.flush();
    			
    		}
    		System.out.println("tmp_Sorted:"+ tmp_Sorted +" map_lineNo2_eachLine.size:"+ map_lineNo2_eachLine.size()
													+" tmp_Sorted.size:"+tmp_Sorted.size());
    		///////.. 
    		TreeMap<Integer, String > map_seq3_line=
    	    		readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
    	    		readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(outputFile_cross_multiAuthID_SORTED, //AuthName!!!AuthID!!!cross-MultiAuthIDmatched!!!distinctCount-cross-MultiAuthIDmatched
    	    																					-1 , //startline, 
    	    																					-1, //endline
    	    																					"debug_label", 
    	    																					false //isPrintSOP
    	    																					);
     		String header=map_seq3_line.get(1);
     		writer_SORTED_addMINauthID.append(header+"!!!ALL_authIDs!!!min_AUTHID"+"\n");
     		String line="";
     		for(Integer _seq:map_seq3_line.keySet()){
     			TreeMap<Integer, Integer> map_currLine_ALLuniqueAuthIDs_as_key=new TreeMap<Integer, Integer>();
     			if(_seq==1) {
     				continue;} //skip header
     			
     			line=map_seq3_line.get(_seq).replace("!!!!!!", "!!!-1!!!");
     			
     			String []s=line.split("!!!");
     			//get all unique auth IDs
     			// "AuthID"
     			map_currLine_ALLuniqueAuthIDs_as_key.put( Integer.valueOf(s[1]) , -1 );
     			//"cross-MultiAuthIDmatched"
     			String []s2=s[2].split("\\$\\$\\$");
     			int cnt=0;
     			///
     			while(cnt<s2.length){
     				
     				if(Integer.valueOf(s2[cnt]) >0){ //omit -1
     					map_currLine_ALLuniqueAuthIDs_as_key.put( Integer.valueOf(s2[cnt]), -1);
     				}
     				cnt++;
     			}
     			//find min AuthID
     			int min = Collections.min(map_currLine_ALLuniqueAuthIDs_as_key.keySet());
     		
     			writer_SORTED_addMINauthID.append( line+ map_currLine_ALLuniqueAuthIDs_as_key.toString().replace("=-1, ", ",").replace("{", "").replace("}", "")
     													  .replace("=-1", "")
     											+"!!!"+min+"\n" );
     			writer_SORTED_addMINauthID.flush();
     		}
    		
		} catch (Exception e) {
			e.printStackTrace();
			// TODO: handle exception
		}
	}
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        String baseFolder="/Users/lenin/Dropbox/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/authornameextract2/";
        	   baseFolder="/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/";
        
        // PRE-REQUISTION: 
        //	(1) run "cleaning_and_convert.java". Output of "readFile_and_remove_stop_words_for_eachLine" will be used for "inFile_1_all_tokens" below.
        //  (2) "map_10_Query_Topics" add the 10 topics to this map
        //	(3) run readFile_eachLine_get_a_particular_token_calc_frequency.calc_Frequency_Of_Token_AND_output_unique_values_in_token 
        //		to get "outputUniqNames_only_authors.txt" and "outputUniqNames_auth_freq_no_duplicates.txt"
        //     *****THIS needs RUNNING "ground_truth.java" - Flag =[1,2] OR Flag==[12]
        //  (4) "outputUniqNames_only_authors.txt" present in "baseFolder" named "ds<N>" from Step (3) above
        //  (5) [Flag_for_Task==4] - Set "is_run_tfidf_based_cosine=true" to calculate tf-idf based cosine similairty, otherwise only freq-based cosine calculated.
        
        // File 1 - BOTH POLICY AND TRAD
        //inFile_1==>first column from this file will be doc_id OR line number of this file will become doc id
        String  inFile_1_all_tokens=baseFolder+"20000.txt";         
        int     inFile_1_token_containing_body_text=1;
        int     inFile_1_token_containing_author_names=4;
        int		inFile_1_token_containing_source_URL= 2;
		int 	index_of_token_of_Source_FileName=7 ; 		//change manual
        String  inFile_1_pattern_to_find_keywords="Keywords:";
        //OBSELETE - NOT USING THIS FILE ACTUALLY
        String  inFile_2_containing_bodyText_only=baseFolder+"mergedall-2016-T9-only-bodyText.txt_ONLY_ENGLISH.txtremoved_NOISE.txt";
        //File 3 having list of unique author names
        //(run readFile_eachLine_get_a_particular_token_calc_frequency.calc_Frequency_Of_Tokencalc_Frequency_Of_Token_AND_output_unique_values_in_token
        String  inFile_3_having_unique_author_name=baseFolder+"uniq_authors.txt"; //"outputUniqNames_only_authors.txt";
        //The below file is manually created .. It has dirty AUTHOR NAMES such as bbc news
        String  inFile_4_having_DIRTY_author_name="/Users/lenin/Dropbox/workspace-sts-3.4.0.RELEASE.macmini/crawler/src/p6/dirty_authNames";
        /////////////////////    Output        //////////////// 
        String outFile_for_vocabulary=baseFolder+"voc.txt";
        ///this is only FREQUENCY.. NO TF-IDF
        String outFile_for_doc_id_and_word_id_with_frequency=baseFolder+"doc_id_word_id_AND_freq__.txt";
        String outFile_for_author_id_and_doc_ids=baseFolder+"auth_id_doc_id.txt"; //
        String outFile_for_author_name_AND_author_id=baseFolder+"authName_AND_auth_id.txt"; //
        String outFile_for_authorNameNOSPACE_AND_authorNameVARIANT=baseFolder+"authNameNOSPACE_AND_authNameVariant.txt";
        String debugFile=baseFolder+"debug-p6_prepare_matrix_author_id_vs_docid_AND_unique_voc_word.txt";
        String YES_CSV_Filter_for_inFile_1=""; //NOT implemented now
        String outFile_auth_id_doc_id_queryTopicRelated=baseFolder+"auth_id_doc_id_queryTopicRelated.txt";
        String output_curr_topic_auth_id_FREQ_OR_tf_idf="";
        String replace_string_for_sourceChannelFilePath="infile:/users/lenin/downloads/#problems/p6/merged_/rawxmlfeeds2csv/merged/all_all_body_url_datetime_auth_keywor_subjec_";
        //if this is true, above created curr_q_topic wise
        //	, each  author and word_id freq, will be converted
        	 boolean is_run_tfidf_based_cosine=true; //Flag_for_Task==4
        	 boolean is_run_freq_based_cosine=false; //Flag_for_Task==4
        	 boolean is_load_vocabulary_from_past_run=false; //Flag_for_Task==1
             int	 Flag_for_Task=4; //set flag
             //**************** Flag_for_Task ****************************************************************
             // 1   = (a) uniq authNames AND (b) matrix_author_id_vs_docid 
             // 11  = remove DIRTY author names such as bbc
             //	2   = (a) generate files <authID,docID,query> AND  (b) TFIDF  
             // 3   = pagerank input files AND gen <authid!!!word_id1:CUMM_FREQ,wor_id_2:CUMM_FREQ>
             // 4   = parse_doc_each_pair_lines_calc_cosine < - use more flags "is_run_*" above
             // 99  = run all 
             // 1112 = run for Flag = 1,11,2 <----
             //************************************************************************************************
             
			 String suffixPR="TFIDF";  // OBSELETE , NOT USED REALLY .FREQ (or) TFIDF 
			 
        TreeMap<Integer, String> map_10_Query_Topics=new TreeMap<Integer, String>();
        map_10_Query_Topics.put(1,"india");map_10_Query_Topics.put(2,"syria");
        map_10_Query_Topics.put(3,"boko haram");map_10_Query_Topics.put(4,"climate change");
        map_10_Query_Topics.put(5,"election");map_10_Query_Topics.put(6,"crime");
//        map_10_Query_Topics.put(6,"cancer");map_10_Query_Topics.put(3, "britain");
//        map_10_Query_Topics.put(9,"trafficking");
//        map_10_Query_Topics.put(4, "kenya");
        
        long t0 = System.nanoTime();
        //////////////////////////  
        if(Flag_for_Task==1 || Flag_for_Task==99 || Flag_for_Task==1112){

	         // STEP 1.1 - get UNIQUE NAME and 
        	String inputFolderName_OR_Single_File = "";   debugFile="";
//			String outputFileName="";
			String outputFileName_onlykeywords="";
			
			//OVERWRITE --- INPUT FOLDER or file
			inputFolderName_OR_Single_File=inFile_1_all_tokens;
			
			//OVERWRITE --  (OUTPUT FILE) only keywords
//			outputFileName=inFile_3_having_unique_author_name; // baseFolder+"uniq_authors.txt";
			outputFileName_onlykeywords=baseFolder+"OnlyAuthors_asc_length.txt";
			int index_of_token_interested_for_freq_count=inFile_1_token_containing_author_names; //change manual
			int index_of_token_of_SourceURL=inFile_1_token_containing_source_URL; //change manual
					debugFile=baseFolder+"debugTooken.txt";
			String  delimiter_4_inputFolderName_OR_Single_File="!!!";
			boolean is_split_on_blank_space_in_token=false;
			boolean is_do_stemming_on_word=false;
			boolean is_split_CSV_on_the_given_token=true;
			boolean is_token_interested__authors=true;
			
			String pathslash="//";
			 
			//this moved in "ground_truth.java" - METHOD: calc_Frequency_Of_Token_AND_output_unique_values_in_token
			readFile_eachLine_get_a_particular_token_calc_frequency.
			calc_Frequency_Of_Token_AND_output_unique_values_in_token
																(
																inputFolderName_OR_Single_File, //INPUT
																delimiter_4_inputFolderName_OR_Single_File,
																inFile_3_having_unique_author_name, //OUTPUT
																false, //is_Append_outputFileName,
																outputFileName_onlykeywords,
																false, //is_Append_outputFileName_onlykeywords,
																inFile_3_having_unique_author_name+"_WORD_DOCUMENT_FREQ.txt",
																index_of_token_interested_for_freq_count, //index_of_token_interested_for_freq_count
																index_of_token_of_SourceURL, //used only is_token_interested__authors==true
																index_of_token_of_Source_FileName, //used only is_token_interested__authors==true
																replace_string_for_sourceChannelFilePath,
																is_token_interested__authors,
																is_split_on_blank_space_in_token, //is_split_on_blank_space_in_token
																is_do_stemming_on_word, //is_do_stemming_on_word
																"", //YES_filter
																"", // NO_FILTER
																true, //  isMac
																is_split_CSV_on_the_given_token, //  is_split_CSV_on_the_given_token
																debugFile,
																true,  //is_add_last_token_of_inputFolderName_to_OUTPUT
																false  // isSOPprint
																);
//			   
//		       String outFile_noDuplicates=baseFolder+"outputUniqNames_auth_DistinctDocs_no_duplicates.txt";
		   
	         // STEP 1.2 - MAIN matrix author_id AND  doc_id (CSV) - below method MAIN one
	         p6_prepare_matrix_author_id_vs_docid_AND_unique_voc_word(
			        		 									baseFolder,
			                                                    inFile_1_all_tokens, //first column from this file will be doc_id OR line number of this file will become doc id
			                                                    inFile_1_token_containing_body_text,// each line has a author name
			                                                    inFile_1_token_containing_author_names,
			                                                    inFile_1_token_containing_source_URL,
			                                                    inFile_1_pattern_to_find_keywords,
			                                                    YES_CSV_Filter_for_inFile_1,
			                                                    inFile_2_containing_bodyText_only,
			                                                    inFile_3_having_unique_author_name,  //UNIQUE Author names
//			                                                    inFile_4_having_DIRTY_author_name, // inFile_4_having_DIRTY_author_name
			                                                    outFile_for_vocabulary,
			                                                    outFile_for_author_name_AND_author_id, //(1)used in read_plsi_output_file_d_z_probability
			                                                    outFile_for_authorNameNOSPACE_AND_authorNameVARIANT,
			                                                    outFile_for_doc_id_and_word_id_with_frequency,
			                                                    outFile_for_author_id_and_doc_ids, //(1)used in read_plsi_output_file_d_z_probability 
			                                                    									// (2) this output is not used in another other methods below
			                                                    outFile_auth_id_doc_id_queryTopicRelated,//this output is used in below further methods
			                                                    map_10_Query_Topics,
			                                                    debugFile,
			                                                    is_load_vocabulary_from_past_run,
			                                                    false // isSOPprint
	                                                    		);
	         
			  // this moved in "ground_truth.java" This file is produced by running "Flag_for_Task==1"
//		       String inputFile_AuthorName_AuthorID=baseFolder+"authName_AND_auth_id.txt";
	           String outFile_noDuplicates=baseFolder+"outputUniqNames_auth_DistinctDocs_no_duplicates.txt";
	            //	STEP 1.3 - remove duplicates in name for p6 problem
		       readFile_eachLine_get_a_particular_token_calc_frequency.
	            p6_remove_duplicates_of_authorName_in_freq_file(
	            												baseFolder,
	            												inFile_3_having_unique_author_name, //authorName!!!Frequency  // input
						            							outFile_for_author_name_AND_author_id, //authName!!!authID (input) produced by running "Flag_for_Task==1"
						            							outFile_noDuplicates
	            												);
	            
	           //note: ds10 database: some authors belongs to many domain->zeina karam
	            	
			System.out.println("output file:"+inFile_3_having_unique_author_name);
			
        }
        
        String outFile_for_author_name_AND_author_id_NO_dirtyAuthNames=outFile_for_author_name_AND_author_id+"_NO_dirtyAuthNames.txt";
        String outFile_for_author_name_AND_author_id_YES_dirtyAuthNames=outFile_for_author_name_AND_author_id+"_YES_dirtyAuthNames.txt";
        
    	String outFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames=outFile_auth_id_doc_id_queryTopicRelated+"_NO_dirtyAuthNames.txt";
    	String outFile_auth_id_doc_id_queryTopicRelated_YES_dirtyAuthNames=outFile_auth_id_doc_id_queryTopicRelated+"_YES_dirtyAuthNames.txt";
        
    	String outFile_for_author_id_and_doc_ids_NO_dirtyAuthNames = outFile_for_author_id_and_doc_ids+"_NO_dirtyAuthNames.txt";// 
    	String outFile_for_author_id_and_doc_ids_YES_dirtyAuthNames = outFile_for_author_id_and_doc_ids+"_YES_dirtyAuthNames.txt";
    	
    	
        if(Flag_for_Task==11  || Flag_for_Task==99|| Flag_for_Task==1112){
        	
        	// remove_DIRTY_authName_from_3Files
        	remove_DIRTY_authName_from_3Files(
        										inFile_4_having_DIRTY_author_name,//INPUT
        										outFile_for_author_name_AND_author_id,//INPUT
        										outFile_auth_id_doc_id_queryTopicRelated,//INPUT
        										outFile_for_author_id_and_doc_ids, //INPUT
        										outFile_for_author_name_AND_author_id_NO_dirtyAuthNames,    //OUTPUT
        										outFile_for_author_name_AND_author_id_YES_dirtyAuthNames,   //OUTPUT
        										outFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames, //OUTPUT 
        										outFile_auth_id_doc_id_queryTopicRelated_YES_dirtyAuthNames, //OUTPUT
        										outFile_for_author_id_and_doc_ids_NO_dirtyAuthNames,//OUTPUT
        										outFile_for_author_id_and_doc_ids_YES_dirtyAuthNames//OUTPUT
        									  );
        	
        }
        
        String outFile_for_author_id_and_word_id_with_TFIDF=baseFolder+"doc_id_word_id_AND_tfidf__.txt";

		String outputFile_cross_multiAuthID=outFile_for_author_name_AND_author_id +"_MAPPEDmultiAuthID.txt";
		String outputFile_cross_multiAuthID_SORTED=outFile_for_author_name_AND_author_id +"_MAPPEDmultiAuthID_SORTED.txt";
		String outputFile_cross_multiAuthID_SORTED_addMINauthID=outFile_for_author_name_AND_author_id +"__MAPPEDmultiAuthID_SORTED_add_minAuthID.txt";
		String outFile_auth_id_doc_id_queryTopicRelated_applying_cross_multiAuthIDs=outFile_auth_id_doc_id_queryTopicRelated+"__applying_multi_crossAuthIDs.txt";
		///////// //////////// 
        if(Flag_for_Task==2  || Flag_for_Task==99 || Flag_for_Task==1112){
        	
        	////  STEP :  CROSS Author Names that belong to same person/entity but has many Auth IDs
        	cross_map_multi_authIDs_of_same_AuthName(
        												baseFolder,
        												outFile_for_author_name_AND_author_id, //input file <----from p6_prepare_matrix_author_id_vs_docid_AND_unique_voc_word
        												outputFile_cross_multiAuthID, //output
        												outputFile_cross_multiAuthID_SORTED, //output
        												outputFile_cross_multiAuthID_SORTED_addMINauthID //output
        											);
        	
        	//normalize_duplicate_authID_applying_cross_multiAuthIDs (out of this method is input to "split_auth_id_doc_id_queryTopic")
        	normalize_duplicate_authID_applying_cross_multiAuthIDs(
        														baseFolder,
        														outputFile_cross_multiAuthID_SORTED_addMINauthID, //input
        														outFile_auth_id_doc_id_queryTopicRelated, //input <----from p6_prepare_matrix_author_id_vs_docid_AND_unique_voc_word
        														outFile_auth_id_doc_id_queryTopicRelated_applying_cross_multiAuthIDs //output
        														);
        	
        	//MARK SOME AUTH IDS based on AUTH NAMES AS DIRTY**
        	 
	        ////////////////////////// STEP: convert "docid_Wordid_freq" to "docid_Wordid_tfidf"
	        convert_doc_word_FREQ_file_to_doc_word_TFIDF.convert_doc_word_FREQ_file_to_doc_word_TFIDF(
	        																	baseFolder, 
	        																	outFile_for_doc_id_and_word_id_with_frequency,
	        																	outFile_for_author_id_and_word_id_with_TFIDF
	        																	);
        	
//	          //////////////////each current query topic has separate <authID, docID>       ////////////////////////////////
//        	  //produces "<<query>_authID_DocID>.txt"
	          split_auth_id_doc_id_queryTopic(
	        		  							baseFolder , 
	        		  							outFile_auth_id_doc_id_queryTopicRelated_applying_cross_multiAuthIDs, //GLOBAL (INPUT) replaced "outFile_auth_id_doc_id_queryTopicRelated,
	        		  						  	map_10_Query_Topics
	        		  						 );
	          
        }
        
         	////////////////// each query topic, get docID////////////////////////////////////////////////
          for(int curr_query_topic:map_10_Query_Topics.keySet()){
        	  //
        	  String curr_q_topic=map_10_Query_Topics.get(curr_query_topic);
			  String outFile_authID_old_new=baseFolder+"authID_old_new_"+curr_q_topic+".txt";
			  String outFile_docID_old_new=baseFolder+"docID_old_new_"+curr_q_topic+"_NOUSE.txt";
			  String outFile_query_auth_id_doc_id_queryTopicRelated=baseFolder+curr_q_topic+"_auth_id_doc_id_queryTopicRelated.txt";
			  //////////// //////////// 
			 if(Flag_for_Task==3  || Flag_for_Task==99 ){
				 
				  System.out.println("curr_q_topic:"+curr_q_topic +" outFile_for_author_id_and_word_id_with_frequency:"+outFile_for_doc_id_and_word_id_with_frequency);
			  
		          //each current query topic has separate <NEW_authID, NEW_docID>
		          generate_NEW_IDs_for_auth_id_AND_doc_id_for_each_queryTopic(
		        		  													baseFolder,
		        		  													outFile_query_auth_id_doc_id_queryTopicRelated, //input
		        		  													outFile_authID_old_new, //output
		        		  													outFile_docID_old_new //output
		        		  													);
	          
	          // FOR EACH of new file from "generate_NEW_IDs_for_auth_id_AND_doc_id_for_each_queryTopic";
	          // file->"outFile_auth_id_doc_id_queryTopicRelated" , get PageRANK
//	          String pageRank_inFile=outFile_auth_id_doc_id_queryTopicRelated+"_pageRank_inFile_NEW_authid.txt";
	          // OBSOELETE*** generate pageRANK
//	          generate_pageRANK_inputFile_with_new_AuthID(
//		        		  									  baseFolder+"_auth_id_doc_id_queryTopicRelated.txt", //<NEW_AUTH_ID, OLD_DOC_ID, Q Topic> for specific Q Topic 
//		        		  									  outFile_authID_old_new,
//		        		  									  pageRank_inFile
//	        		  									  );
		          String input_authID_docID_queryTopicRelated=baseFolder+curr_q_topic+"_auth_id_doc_id_queryTopicRelated.txt";
		          String out_topicWISE_authID_docID_CSV=baseFolder+curr_q_topic+"_authID_docIDCSV__.txt";
		          System.out.println("out_topicWISE_authID_docID_CSV:"+out_topicWISE_authID_docID_CSV);
	          	////////////////////////////////////////////////////////////////////////////////////////////////
	          //NEEDED--  This not calculating COSINE ---> convert to <<black>!!!auth_ID!!!Doc_ID_CSV>
		          generate_pageRANK_INPUT_inputFile_each_auth_pair_COSINE(
		        		  											baseFolder,
		        		  											input_authID_docID_queryTopicRelated,  //INPUT
		        		  											outFile_for_doc_id_and_word_id_with_frequency, //global /// INPUT
		        		  											out_topicWISE_authID_docID_CSV //OUTPUT
		        		  											);
	
	          	//////////////////////////////////////////////////////////////////////////////////////////////
		          // below is actually only FREQUENCY, want TF-IDF override with "is_override_TFIDF_over_freq" variable below
		          output_curr_topic_auth_id_FREQ_OR_tf_idf=baseFolder+"tf-idf-for-"+curr_q_topic+".txt";
		          String outputFile_docID_wordID_freq=baseFolder+"doc_id_word_id_AND_freq__.txt";

	          	// IMPORTANT: below method only handles frequency, TF-IDF is not handled as it is complicated.
	            // So output of below method pass to ""convert_doc_word_FREQ_file_to_doc_word_TFIDF"
	          	// convert to <authid!!!word_id1:CUMM_FREQ,wor_id_2:CUMM_FREQ>
		          p6_evaluation_pageRank.convert_each_author_each_doc_FREQ_to_each_auth_CUMM_FREQ(
																		   baseFolder,
																		   outputFile_docID_wordID_freq,// inFile_1_for_each_doc_word_id_AND_tf_idf, // freq 
																		   out_topicWISE_authID_docID_CSV, // inFile_2_for_each_auth_id_AND_doc_id_CSV
																		   output_curr_topic_auth_id_FREQ_OR_tf_idf, //output File
																		   false  //isAppendoutFile
								   									       );
				
			  } //if(Flag_for_Task==3){
			  
	          String curr_topic_auth_id_FREQ_OR_tf_idf_ACTUAL=baseFolder+"tf-idf-for-"+curr_q_topic+".txt";

	          if(Flag_for_Task==4  || Flag_for_Task==99){
	          	
	          	if(is_run_tfidf_based_cosine){
		          // Below method convert CUMM FREQ	 to CUMM TFIDF//////////////////////
	          		////**REDUNDENCY-. this is already done (convert_doc_word_FREQ_file_to_doc_word_TFIDF)
//					convert_doc_word_FREQ_file_to_doc_word_TFIDF.convert_doc_word_FREQ_file_to_doc_word_TFIDF(
//																  baseFolder
//															    , output_curr_topic_auth_id_FREQ_OR_tf_idf //input FREQ
//																, curr_topic_auth_id_FREQ_OR_tf_idf_ACTUAL //OUPUT TF-IDF
//																);
					
					
	          	} //if(is_run_tfidf_based_cosine){

	          //generate_pageRANK_INPUT_inputFile_each_auth_pair_COSINE creates same file name as below.
	          String t_pagerank=output_curr_topic_auth_id_FREQ_OR_tf_idf+"_PAGERANK_"+"FREQ"+".txt";
	          
	          //if(curr_topic_auth_id_tf_idf.indexOf("boko haram")==-1) continue; 
	          
	          // parse_doc_each_pair_lines_calc_cosine (frequency based)
	          // SLOWER=16 m**** FREQ based cosine similarity calc
	          if(is_run_freq_based_cosine){
	        	  t_pagerank=output_curr_topic_auth_id_FREQ_OR_tf_idf+"_PAGERANK_"+"FREQ"+".txt"; //OVERRIDE
		          DocumentParser_AND_Cosine.parse_doc_each_pair_lines_calc_cosine(
			        		  													baseFolder, 
			        		  													output_curr_topic_auth_id_FREQ_OR_tf_idf, //inputFile FREQ
			        		  													t_pagerank, //outputFile <>
			        		  													false, //isAppend_outFile
			        		  													-1, 
			        		  													-1,
			        		  													false //false=dont check on past ran
			        		  													);
		          
		          // output cosine <new_authID_1,new_authID_2,cosine_score>
		          String t_pagerank_new=output_curr_topic_auth_id_FREQ_OR_tf_idf+"_PAGERANK_new_AuthID.txt";
		          //mapping of old_authID and new_authID
		          String t_pagerank_mapping=output_curr_topic_auth_id_FREQ_OR_tf_idf+"_PAGERANK_old_AuthID_new_AuthID_map.txt";
		          // convert old_doc_id to new_doc_id where new has consecutive sequence .
		          input_authID_pair_out_unique_authID_for_inFile_of_COSINE(
		        		  													baseFolder,
		        		  													t_pagerank, //INPUT
		        		  													t_pagerank_new, //output cosine <new_authID_1,new_authID_2,cosine_score>
		        		  													t_pagerank_mapping //output
		        		  													);
		          
	          } //if(is_run_freq_based_cosine){
	          
	          	// TF-IDF based cosine similarity calc
	          	if(is_run_tfidf_based_cosine){
	              //Override
	  	          t_pagerank=curr_topic_auth_id_FREQ_OR_tf_idf_ACTUAL+"_PAGERANK_"+"TFIDF"+".txt";
		          // parse_doc_each_pair_lines_calc_cosine (tf-idf based)
		          DocumentParser_AND_Cosine.parse_doc_each_pair_lines_calc_cosine(
																			baseFolder, 
																			curr_topic_auth_id_FREQ_OR_tf_idf_ACTUAL, //inputFile TF-IDF
																			t_pagerank, //outputFile <>
																			false, //isAppend_outFile
																			-1, 
																			-1,
																			false //false=dont check on past ran
																			);
		          
		          // output cosine <new_authID_1,new_authID_2,cosine_score>
		            String t_pagerank_new=curr_topic_auth_id_FREQ_OR_tf_idf_ACTUAL+"_PAGERANK_new_AuthID.txt";
		          //mapping of old_authID and new_authID
		            String t_pagerank_mapping=curr_topic_auth_id_FREQ_OR_tf_idf_ACTUAL+"_PAGERANK_old_AuthID_new_AuthID_map.txt";
		          // convert old_doc_id to new_doc_id where new has consecutive sequence .
		          input_authID_pair_out_unique_authID_for_inFile_of_COSINE(
		        		  													baseFolder,
		        		  													t_pagerank, //INPUT
		        		  													t_pagerank_new, //output cosine <new_authID_1,new_authID_2,cosine_score>
		        		  													t_pagerank_mapping //output
		        		  													);
		          
	          	}// if(is_run_tfidf_based_cosine){
	          
	          } //if(Flag_for_Task==4){
	          	
          } // END for(int curr_query_topic:map_10_Query_Topics.keySet()){
          
          // 
          for(int curr_query_topic:map_10_Query_Topics.keySet()){
        	  String curr_q_topic=map_10_Query_Topics.get(curr_query_topic);
          }

      	 System.out.println("Time Taken:" 
		         			 	+ NANOSECONDS.toSeconds(System.nanoTime() - t0) +" seconds; "
					       	   	+ (NANOSECONDS.toSeconds(System.nanoTime() - t0))/60 +" minutes" );
         
         // OBSOLETE OVERWRITE 
//       remove_whiteSpace_in_voc(outFile_for_vocabulary_DUMMY, //INPUT
//       						 outFile_for_vocabulary+"_noWHITESPACE.txt" //OUTPUT
//       						);
      	 
     	//OBSOLETE(/////////////////////////////REMOVE STOP WORDS (word id) from the 
         //(1) freq file--outFile_for_author_id_and_word_id_with_frequency (2) tf-idf file--outFile_for_author_id_and_word_id_with_TFIDF
         String outputFile_removedStopWordID=outFile_for_doc_id_and_word_id_with_frequency+"REMOVE_stopWordID.txt";
         	// freq-based file - remove stop word id
//         	remove_stopWordID_from_authID_wordID_freq(
//         											  baseFolder,
//         											  outFile_for_author_id_and_word_id_with_frequency,
//         											  outFile_for_vocabulary,
//         											  outputFile_removedStopWordID
//         											 );
         	  outputFile_removedStopWordID=outFile_for_author_id_and_word_id_with_TFIDF+"REMOVE_stopWordID.txt";
         	// TFIDF-based file - remove stop word id
//         	remove_stopWordID_from_authID_wordID_freq(
//         											  baseFolder,
//         											  outFile_for_author_id_and_word_id_with_TFIDF,
//         											  outFile_for_vocabulary,
//         											  outputFile_removedStopWordID
//         											 );
          
          }

}
