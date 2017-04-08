
import static java.util.concurrent.TimeUnit.NANOSECONDS;

import crawler.find_Verb_AND_Noun_JustBfr_from_TaggedString_inFile;
import crawler.find_domain_name_from_Given_URL;

import crawler.clean_retain_only_alpha_numeric_characters;
import crawler.calc_checksum;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.StringWriter;
import java.sql.Clob;
import java.util.HashMap;
import java.util.TreeMap;

import convert.convertPatternStringToMap;
import crawler.LoadMultipleValueToMap2_AS_KEY_VALUE;
import crawler.convert_mapString_to_TreeMap;
import crawler.crawler;
import crawler.find_Language_from_String;
import crawler.isInteger;
import crawler.load_domainname_pattern4crawledHTML;
import crawler.readFile_eachLine_get_a_particular_token_calc_frequency;
import crawler.readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile;
import crawler.readFile_pick_a_Token_doNLPtagging_writeOUT;
import crawler.readFile_verify_and_Correct_No_Of_Tokens_In_EachLine;
import crawler.removeUnicodeChar;
import crawler.sort_by_datetime_for_extracted_CSV_news_articles;
import crawler.sort_given_treemap;
import crawler.stopwords;
import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreLabel;
import opennlp.tools.cmdline.postag.POSModelLoader;
import opennlp.tools.parser.ParserModel;
import opennlp.tools.postag.POSModel;
import p18.clean_htmlcrawled_remove_htmlTagsETC;

import crawler.readFile_2_Files_match_by_common_primary_key_and_add_a_tokenFrom_SecondFile_to_FirstFile;

public class cleaning_and_convert {

	// Methods
	//----------
	// get_title() 
	// t2_english_only
	// get_all_text_features_cleaned  ( mallet input file)
	// find_get_a_token_and_write_to_another
	// readFile_and_get_only_numbers_from_eachLine
	// load_AuthID_and_AuthName
	// conv_2_mallet_format  ( mallet input file)
	// ** Prepares MALLET input file for LDA
	//----------
	
	//remove_NOISE
	private static void remove_NOISE(String outFile_only_int_SetOftoken, 
									 String outFile_all ) {
		// TODO Auto-generated method stub
		try {
			
//			BufferedReader reader=null;
//			BufferedReader reader_all=null;
//			 
//				reader = new BufferedReader(new FileReader(outFile_only_int_SetOftoken));
//				reader_all = new BufferedReader(new FileReader(outFile_all));
				
				FileWriter writer_onlyBODYTEXT_removed_noise= new FileWriter(new File(outFile_only_int_SetOftoken+"removed_NOISE.txt" ));
				FileWriter writer_alltoken_remove_noise= new FileWriter(new File(outFile_all +"_removed_NOISE.txt"));
				
				FileWriter writer_onlyBODYTEXT_only_noise= new FileWriter(new File(outFile_only_int_SetOftoken+"only_NOISE.txt" ));
				FileWriter writer_alltoken_only_noise= new FileWriter(new File(outFile_all +"_only_NOISE.txt"));
			
				
				TreeMap<Integer,String> map_inputFile_only=
						  readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
									 .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
																 							 outFile_only_int_SetOftoken, 
																						 	 -1, //startline, 
																							 -1, //endline,
																							 " reloading..", //debug_label
																							 false //isPrintSOP
																							 );
				
				TreeMap<Integer,String> map_inputFile_all=
						  readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
						 .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
								 							 outFile_all, 
														 	 -1, //startline, 
															 -1, //endline,
															 " reloading..", //debug_label
															 false //isPrintSOP
															 );
				
				// only 1 token = bodytext
				for(int seq: map_inputFile_only.keySet()){
					if(map_inputFile_only.get( seq).toLowerCase().indexOf("([" )>=0 || map_inputFile_only.get( seq).toLowerCase().indexOf("(^" ) >=0){ 
						writer_onlyBODYTEXT_only_noise.append(map_inputFile_only.get(seq) +"\n");
						writer_onlyBODYTEXT_only_noise.flush();
					}
					else{
						writer_onlyBODYTEXT_removed_noise.append(map_inputFile_only.get(seq) +"\n");
						writer_onlyBODYTEXT_only_noise.flush();
					}
				}
				//all token
				for(int seq: map_inputFile_all.keySet()){
					if(map_inputFile_all.get( seq).toLowerCase().indexOf("([" )>=0 || map_inputFile_all.get( seq).toLowerCase().indexOf("(^" ) >=0){ 
						writer_alltoken_only_noise.append(map_inputFile_all.get(seq) +"\n");
						writer_alltoken_only_noise.flush();
					}
					else{
						writer_alltoken_remove_noise.append(map_inputFile_all.get(seq) +"\n");
						writer_alltoken_remove_noise.flush();
					}
				}
				 
			
		} catch (Exception e) {
			e.printStackTrace( );
			// TODO: handle exception
		}
		
	}
	
	// clean_url_second_File
	private static void clean_url_second_File(String second_File,
											  String outputFile
											 ) {
		// TODO Auto-generated method stub
		try{
			FileWriter writer=new FileWriter(new File(outputFile));
			TreeMap<Integer, String>  map_seq_eachLine=readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
								  readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line
													 		 (second_File
															 ,-1
															 ,-1
															 , "" //debug_label
															 , false // isPrintSOP
															 );
			// 
			for(int seq:map_seq_eachLine.keySet()){
				String currLine=map_seq_eachLine.get(seq);
				String [] arr_tokens=currLine.split("!!!");
				String url= arr_tokens[0];
				/// 
				if(url.indexOf("orig_sourceUrlString:")>=0 ){
					url =url.substring( url.indexOf("orig_sourceUrlString:")+"orig_sourceUrlString:".length() ,
										url.length());
				}
				// 
				writer.append(url+"!!!"+arr_tokens[1]+"\n");
				writer.flush();
			}
			
			
		}
		catch (Exception e){
			e.printStackTrace();
		}
	}
	
	// 
	public static void test(){
		try{
			System.out.println("tes");
			System.out.println("tes");
			System.out.println("tes");
			System.out.println("tes");
			
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}
	//  do_clean_crawling_docs_using_domainNAMES_patternFORNEWSBODYSTART
	private static void do_clean_crawling_docs_using_domainNAMES_patternFORNEWSBODYSTART(
															  		String input_crawled_file, // IN 
															  		String file4_DomainName_NewsStartPatterns, //IN
															  		String output_crawled_file //OUT
																	) {
		// TODO Auto-generated method stub
		try { 
			
				String line = "";
				BufferedReader reader = null;
				reader = new BufferedReader(new FileReader(input_crawled_file));
				FileWriter writer = new FileWriter(output_crawled_file);
				HashMap<String, String> map_uniqueDomainFromINPUTCrawledFile=new HashMap<String, String>();
				
				FileWriter writer_domainName = new FileWriter(output_crawled_file+"_only_DOMAINname_FROM_INPUTcrawledFILE_2_DEBUG.txt");
				
				FileWriter writer_NOglobalPATTERNavailable = new FileWriter(output_crawled_file+"_NOglobalPATTERNavailable .txt");
				// LOADING
				TreeMap<String, String> map_domainName_NewsStartPattern=
									load_domainname_pattern4crawledHTML.load_domainname_pattern4crawledHTML(file4_DomainName_NewsStartPatterns);
				int begin=-1,no_hit_cnt=-1, hit_cnt=-1; String tmp="";
				int lineNo=0;
				//
				while ((line = reader.readLine()) != null) {
					lineNo++;
					
					String [] arr_line=line.split("!!!");
					if(arr_line.length<2){ continue;}
					
					System.out.println("lineNo:"+lineNo+" arr_line[0]:"+arr_line[0]);
					
					String curr_url = arr_line[0];
					try{
//					 if(curr_url.indexOf("http") >=0 ) 
//						 curr_url=curr_url.substring(curr_url.indexOf("orig_sourceUrlString:") +"orig_sourceUrlString:".length(),
//								 						curr_url.length());
					}
					catch(Exception e){
						continue;
					}
					//CALLING domain name
					String currDomainName=find_domain_name_from_Given_URL.find_domain_name_from_Given_URL(curr_url, false);
					 
					String curr_pattern_4_crawledHTML=map_domainName_NewsStartPattern.get(currDomainName);
					// only domain name
					if(map_uniqueDomainFromINPUTCrawledFile.containsKey(currDomainName)){
						writer_domainName.append(currDomainName+"\n");
						writer_domainName.flush();
					}
					map_uniqueDomainFromINPUTCrawledFile.put(currDomainName, "");
					
					String currBodyText_Crawled=arr_line[1];
					//
					if(curr_pattern_4_crawledHTML==null){
						 writer_NOglobalPATTERNavailable.append(line+"\n" );
						 writer_NOglobalPATTERNavailable.flush();
						 continue;
					}
					
					///////
					//example: if given curr_pattern_4_crawledHTML-->class="article-body";<p> , first search for class, then followed by <p>
					if(curr_pattern_4_crawledHTML.indexOf(";")==-1){
						begin=currBodyText_Crawled.indexOf(curr_pattern_4_crawledHTML);
					}
					else{
						String [] pattern=curr_pattern_4_crawledHTML.split(";");
						begin=currBodyText_Crawled.indexOf(pattern[0]);
						begin=currBodyText_Crawled.indexOf(pattern[1],begin+2);
					}
					
					
//					writerDebug.append("\n NOW** FOUND - currDomainName="+currDomainName 
//												+" begin="+begin+" "+currBodyText_Crawled.contains( curr_pattern_4_crawledHTML) );
					
//					if(currDomainName.equalsIgnoreCase("armenianow.com")){
//						writerDebug.append("\n 	currBodyText_Crawled:"+currBodyText_Crawled);
//					}
					
					if(begin>=0){hit_cnt++;}
					else{ no_hit_cnt++;}
					
					
					
					//
//					if(currBodyText_Crawled.lastIndexOf(currBodyText_rssFeeed_compressed_HALF)>=0){
//						count_half_match++;
//						begin=currBodyText_Crawled.lastIndexOf(currBodyText_rssFeeed_compressed_HALF);
//					}
					
					//
	 
						try{
							tmp=currBodyText_Crawled.substring(begin, begin+9000);
						}
						catch(Exception e){
							 try{
								 tmp=currBodyText_Crawled.substring(begin, begin+7500);
								 System.out.println("ERROR ON >2500");

							 }
							 catch(Exception e2){
							
								try{
									tmp=currBodyText_Crawled.substring(begin,begin+4500);
								}
								catch(Exception e22){
										try{
											tmp=currBodyText_Crawled.substring(begin,begin+3500);
										}
										catch(Exception e3){
											try {
												tmp=currBodyText_Crawled.substring(begin,begin+2500);	
											} catch (Exception e4) {
												// TODO: handle exception
												try {
													tmp=currBodyText_Crawled.substring(begin,begin+1500);	
												} catch (Exception e5) {
													try {
														// TODO: handle exception
														tmp=currBodyText_Crawled.substring(begin,begin+500);											
													} catch (Exception e6) {
														// TODO: handle exception
														tmp="(*(ERROR()*";
													}
			
												}
											}
											
										}
									}	 
							 }
						} //END CATCH

//					System.out.println("curr_pattern:"+curr_pattern_4_crawledHTML
//										+" tmp:"+tmp
//										);

					/// remove HTML TAGS
					tmp=crawler.SubRemoveTagsFromHTML(tmp);
					/// remove junk words
					tmp=clean_htmlcrawled_remove_htmlTagsETC.remove_htmlTagsETC(tmp);
					tmp=tmp.replaceAll("\\s+", " ");
					
					tmp=tmp+" beginINDEXis"+begin;
					//
					if(tmp.indexOf("beginINDEXis-1")==-1){					
						// WRITING
						writer.append(curr_url+"!!!"+tmp+"\n");
						writer.flush();
					}
			    
				} // END while ((line = reader.readLine()) != null) {
//			TreeMap<Integer,String> map_inputFile_all_unique_matched_lines_on_allQUERIES_merge_add2MOREfeature=
//					  readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
//					 .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
//							 							 all_unique_matched_lines_on_allQUERIES_merge_add2MOREfeature, 
//													 	 -1, //startline, 
//														 -1, //endline,
//														 " reloading..", //debug_label
//														 false //isPrintSOP
//														 );
			 
//			for(int seq:map_inputFile_all_unique_matched_lines_on_allQUERIES_merge_add2MOREfeature.keySet()){
//				String currLine=map_inputFile_all_unique_matched_lines_on_allQUERIES_merge_add2MOREfeature.get(seq);
//				
//				
//			}
			

			//bodyText!!!url!!!dateTime!!!author!!!keywords!!!Subject!!!from File!!!bodyText no stopwords!!!docID!!!missing_keywords
			System.out.println("map_domainName_NewsStartPattern:"+map_domainName_NewsStartPattern);
			
		} catch (Exception e) {
			e.printStackTrace();
			// TODO: handle exception
		}
	}

	// load authName and auth_id
	public static  TreeMap<String, Integer>  load_AuthName_and_Auth_id(String inFile,
																		boolean is_normalize_name
																		){
		String line="";
		TreeMap<String, Integer> map=new TreeMap();
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(
												new FileInputStream(inFile)));

			//each line
			while ((line = reader.readLine()) != null) {
				String [] s=line.split("!!!");
				
				
				map.put(  p6_MAIN_Work.p6_normalize_auth_name(s[0]).replace("auth:from: ", ""),
						new Integer(s[1]));
			}// end while
		}
		catch(Exception e){
			e.printStackTrace();
		}
		return map;
	}
	// load auth_id and  authName 
	public static  TreeMap<Integer, String>  load_AuthID_and_AuthName(String inFile,
																	  boolean is_normalize_name){
		String line="";
		TreeMap<Integer, String> map=new TreeMap();
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(
												new FileInputStream(inFile)));

			//each line
			while ((line = reader.readLine()) != null) {
				String [] s=line.split("!!!");
				map.put(new Integer(s[1]),p6_MAIN_Work.p6_normalize_auth_name(s[0]).replace("auth:from: ", ""));
			}// end while
		}
		catch(Exception e){
			e.printStackTrace();
		}
		return map;
	}
	
	// find_get_a_token_and_write_to_another
	public static void find_and_filter_Given_10_topics( String inputFile, 
													    String outFile_1, //get all tokens
													    String outFile_2, //get only bodyText
														int    interested_token_for_bodyText_inputFile){
		BufferedReader br=null;
		try{
			br = new BufferedReader(new FileReader(inputFile));
			FileWriter writer= new FileWriter(new File(outFile_1));
			FileWriter writer_only_bodyText= new FileWriter(new File(outFile_2));
			String line=""; int lineNumber=0; String nline="";
			while( (line = br.readLine()) !=null){
				String original_line=line;
				line=line.toLowerCase();
                lineNumber++;
                System.out.println("lineNo:"+lineNumber);
                 //
//	             if(line.indexOf("afghanistan")==-1 && line.indexOf("china")==-1 &&line.indexOf("iran")==-1
//		             &&line.indexOf("saudi arabia")==-1&&line.indexOf("syria")==-1&&line.indexOf("india")==-1 
//		             && line.indexOf("bangladesh")==-1 &&line.indexOf("russia")==-1&&line.indexOf("pakistan")==-1
//		             &&line.indexOf("egypt")==-1){
//	                 continue;
//	             if(bodyText.indexOf("india")==-1 && bodyText.indexOf("china")==-1 &&bodyText.indexOf("syria")==-1
//			             &&bodyText.indexOf("united states")==-1&&bodyText.indexOf("poverty")==-1&&bodyText.indexOf("ebola")==-1 
//			             &&bodyText.indexOf("pollution")==-1 &&bodyText.indexOf("swine flu")==-1&&bodyText.indexOf("earth quake")==-1
//			             &&bodyText.indexOf("corruption")==-1){
//	             if(bodyText.indexOf("india")==-1 && bodyText.indexOf("china")==-1 &&bodyText.indexOf("syria")==-1
//			             &&bodyText.indexOf("united states")==-1&&bodyText.indexOf("ebola")==-1&&bodyText.indexOf("climate changes")==-1 
//			             &&bodyText.indexOf("cancer")==-1 &&bodyText.indexOf("economy")==-1&&bodyText.indexOf(" hiv ")==-1
//			             &&bodyText.indexOf("crime")==-1){

            	 String []s=original_line.split("!!!");
            	 
            	 
            	 String bodyText=s[interested_token_for_bodyText_inputFile];
            	 bodyText=remove_commas_etc(removeUnicodeChar.removeUnicodeChar(
			    						   remove_custom_Tags(bodyText)));
            	 
            	 if(bodyText.indexOf(" console ")>=0 || bodyText.indexOf("console.")>0
            			 ) //junk js content
            		 continue;
            	 
            	 //t6 - India, Syria, United States, Internet Freedom, Afghanistan, Climate Change, Cancer, Economy, Crime, Abortion law
                // India, China, Syria, United States, Ebola, Climate Change, Cancer, Economy, HIV, Crime
            	 // T5- India, Syria, United States, Internet Freedom, Ebola, Climate Change, Cancer, Economy, Crime, Abortion law
	             if(bodyText.indexOf("india")==-1 && bodyText.indexOf("syria")==-1 &&bodyText.indexOf("united states")==-1
			             &&bodyText.indexOf("internet freedom")==-1&&bodyText.indexOf("australia")==-1&&bodyText.indexOf("climate change")==-1 
			             &&bodyText.indexOf("cancer")==-1 &&bodyText.indexOf("economy")==-1&&bodyText.indexOf("crime")==-1
			             &&bodyText.indexOf("trafficking")==-1){
		                 continue;
		             }
	             else{
	           
	            	 writer.append(original_line+"\n");
	            	 writer.flush();
	            	 
	            	 writer_only_bodyText.append(bodyText+"\n");
	            	 writer_only_bodyText.flush();
	            	 
	             }

			}
			
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}	
		
	// find_get_a_token_and_write_to_another (flag==1)
	public static boolean find_get_a_token_match_on_N_Topics_and_write_to_another(	 String  inputFile,
																				 String  outFile_all_tokens,
																			     String  outFile_int_token,
																			     int     interested_token,
																			     String  q_topic_10_CSV_string,
																			     String  q_topic_10_CSV_string_NOT,
																			     boolean isSOPprint
																			     ){
		BufferedReader br=null;
		int total_matched_lines=0;
		try{
			br = new BufferedReader(new FileReader(inputFile));
			FileWriter writer= new FileWriter(new File(outFile_int_token));
			FileWriter writer_all_token= new FileWriter(new File(outFile_all_tokens));
			String line=""; int lineNumber=0; String nline="";
			String []arr_q_topic=q_topic_10_CSV_string.split(",");
			String []arr_q_topic_NOT=q_topic_10_CSV_string_NOT.split(",");
			
			if(arr_q_topic.length!=arr_q_topic_NOT.length){
				System.out.println("***ERROR: not equal length arr_q_topic.length!=arr_q_topic_NOT.length "+arr_q_topic.length+"--"+arr_q_topic_NOT.length);
				return false;
			}
			
			TreeMap<Integer,String> map_seq_query=new TreeMap<Integer, String>();
			int c=0;
			while(c<arr_q_topic.length){ //copy to map for easy 
				map_seq_query.put(0, arr_q_topic[c]);
				c++;}
			
			//
			while( (line = br.readLine()) !=null){
				String origLine=line;
			 
				line=line.toLowerCase();
                lineNumber++;
                boolean isCurrLine_has_any_one_query=false;
                //check_isCurrLine_has_any_one_query
                isCurrLine_has_any_one_query=check_isCurrLine_has_any_one_query( line, map_seq_query);
                                
                System.out.println("lineno:"+lineNumber);
                //if none of the query in queryCSV found, skip
                if(isCurrLine_has_any_one_query==false){
                	System.out.println("continue as not found in any tokens "+lineNumber);
                    continue;
                } 
                //
				if(line.length()>=1){
					if(line.indexOf("eventobj")>=0) continue;
                    String []s=origLine.split("!!!");
                    if(s.length<interested_token) {
                    	continue;
                    }
                    
                    nline=s[interested_token-1].replace("t:head:", " ").replace("t:", " ").replace("bbody::b ", " ")
    						.replace("bbody::", " ");
                    
                    //
//                    int c2=0;
//                    boolean is_skip_on_curr_query_NOT=false;
//                    while(c2<arr_q_topic.length){
//                    	String curr_query=arr_q_topic[c2];
//                    	String curr_query_NOT=arr_q_topic_NOT[c2];
//                    	//
//                    	if(!curr_query_NOT.equals("dummy")){
//	                    	// if curr_query=india and curr_query_NOT=indiana then skip..
//	                    	if(nline.toLowerCase().indexOf(curr_query) >=0 && 
//	                    			nline.toLowerCase().indexOf(curr_query_NOT) >=0  ){
//	                    		is_skip_on_curr_query_NOT=true;
//	                    		break;
//	                    	}
//                    	}
//                    	c2++;
//                    }
//                    
//                    if(is_skip_on_curr_query_NOT==true) {continue;}
                    
                    if(isSOPprint)
                    	System.out.println("int token:"+nline);
					//writer.append( line.substring(1, line.length()) +"\n");
					writer.append( nline +"\n");
					writer.flush();
					
					writer_all_token.append( origLine +"\n");
					writer_all_token.flush();
					total_matched_lines++;
				}
			}
			
			System.out.println("COMPLETED...total_matched_lines:"+total_matched_lines);
			
		}
		catch(Exception e){
			e.printStackTrace();
		}
		return true;
	}
	
	//
    public static boolean check_isCurrLine_has_any_one_query(String line, TreeMap<Integer, String> map_seq_query) {
		// TODO Auto-generated method stub
		try {
			
			for(int seq:map_seq_query.keySet()){
				
				if(line.toLowerCase().indexOf(map_seq_query.get(seq)) >=0 ){
					return true;
				}
				
			}
			
			
		} catch (Exception e) {
			e.printStackTrace();
			// TODO: handle exception
		}
		return false;
	}

	//get_all_text_features_cleaned ( mallet input file)
	public static void get_all_text_features_cleaned(String inputFile, String outFile ){
		BufferedReader br=null;
		try{
			br = new BufferedReader(new FileReader(inputFile));
			FileWriter writer= new FileWriter(new File(outFile));
			String line=""; int lineNumber=0; String nline="";
			while( (line = br.readLine()) !=null){
				line=line.toLowerCase();
                                lineNumber++;
  
				if(line.length()>=1){
                                        String keyword="";
                                        String []s=line.split("!!!");
                                        String source_token2="";
                                        
                                        if(s[1].indexOf("-rnd")>=0)
                                            source_token2=" ";
                                        else
                                            source_token2=s[1];
                                                
                                        if(s[2].indexOf("err:")>=0)
                                            keyword=" ";
                                        else
                                            keyword=s[2];
                                        
                                        nline=lineNumber+"\t"+"A"+"\t"+source_token2+" "+keyword+" "+s[6]+" "+s[8];
                                        nline=nline.replace("auth:from::from:", "").replace("keywords:keywords::keywords:", " ")
			                                        .replace("t:head:", " ").replace("bt:body::", " ").replace("auth:from:","").replace(" bt:", " ")
			                                        .replace("emm india"," ").replace("latest headlines"," ").replace("indian headlines", " ")
			                                        .replace("feeds.reuters.com-rnd", " ")
			                                        .replace("  ", " ").replace("  ", " ");
                                        
                                        System.out.println("s:"+s[8]);
                                        System.out.println("l:"+nline);
										//writer.append( line.substring(1, line.length()) +"\n");
										writer.append("\n"+ nline );
				}
				
				writer.flush();
			}
			
			System.out.println("tes");
			
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}
	
	//convert_mapValue_get_only_values_as_String
	public static String convert_mapValue_get_only_values_as_String(String inString){
		
		TreeMap<String, String> map_= new TreeMap<String, String>();
		String out="";
		try {
			map_=convert_mapString_to_TreeMap.convert_mapString_to_SS_TreeMap(inString,"=",false );
			//get concatenate values with " "
			for(String key:map_.keySet()){
				if(out.length()==0)
					out=map_.get(key);
				else
					out=out+" "+map_.get(key);
			}
			
		} catch (Exception e) {
			// TODO: handle exception
		}
		return out;
	}
	// is_given_authName_a_dirtyOne
	public static boolean is_given_authName_a_dirtyOne(
														String authName,
														TreeMap<Integer,String> map_eachLine_dirtyAuthName,
														String 	folder_dirtyAuthorName,
														String  eachLine,
														boolean is_write_to_outputFile_dirtyAuthName
													){
		authName=authName.toLowerCase();
		
		try{
			for(int seq_:map_eachLine_dirtyAuthName.keySet()){
				String curr_DIRTYAuth=map_eachLine_dirtyAuthName.get(seq_).toLowerCase();
				
				int idx_begin_from_colon=eachLine.toLowerCase().indexOf("from:");
				int idx_begin_currAuth=authName.indexOf(curr_DIRTYAuth);
				int idx_begin_currAuth_in_eachLine=eachLine.toLowerCase().indexOf(curr_DIRTYAuth, idx_begin_from_colon+5);
				
				int end_index_of_currLine_authName=eachLine.indexOf("!!!", idx_begin_from_colon+5 );
				String currLine_authName=eachLine.toLowerCase().substring(idx_begin_from_colon+5 , end_index_of_currLine_authName);
				
				
				//
				if(curr_DIRTYAuth.toLowerCase().indexOf(".line")>=0){
					if(eachLine.toLowerCase().indexOf(curr_DIRTYAuth.toLowerCase().replace(".line", "")) >=0  )
						return true;
				}
				
				/// 
				if(curr_DIRTYAuth.indexOf(".equal")>=0){
					
					if(currLine_authName.replace(" ", "").equals( curr_DIRTYAuth.replace(".equal", "").replace(" ", "") )){
						
						String outFile_for_curr_dirtyAuthName=folder_dirtyAuthorName+curr_DIRTYAuth.replace(" ", "") +".txt";
						FileWriter writer_dirtyAuthNAme=new FileWriter(outFile_for_curr_dirtyAuthName, true);
						
						if(is_write_to_outputFile_dirtyAuthName){
							writer_dirtyAuthNAme.append(eachLine+"\n");
							writer_dirtyAuthNAme.flush();
							
							//below only debug
//							writer_dirtyAuthNAme.append("debug 2 authName:"+authName+" currAuth:"+currAuth+" "+authName.indexOf(currAuth)
//														+" "+idx_begin_currAuth_in_eachLine+ " "+idx_begin_from_colon +" "+(idx_begin_currAuth_in_eachLine - idx_begin_from_colon)
//														+" "+idx_begin_currAuth+" eachLine:"+eachLine+"\n"
//														);
							
						}
						writer_dirtyAuthNAme.close();
						 
						return true;
						
					}
					
				}
				else if(	idx_begin_currAuth >= 0 &&
					(idx_begin_currAuth_in_eachLine - idx_begin_from_colon) <=6   //author NAme should not be far away from from: 
				  ){
					
					String outFile_for_curr_dirtyAuthName=folder_dirtyAuthorName+curr_DIRTYAuth.replace(" ", "") +".txt";
					FileWriter writer_dirtyAuthNAme=new FileWriter(outFile_for_curr_dirtyAuthName, true);
					
					if(is_write_to_outputFile_dirtyAuthName){
						writer_dirtyAuthNAme.append(eachLine+"\n");
						writer_dirtyAuthNAme.flush();
						
						//below only debug
//						writer_dirtyAuthNAme.append("debug authName:"+authName+" currAuth:"+currAuth+" "+authName.indexOf(currAuth)
//													+" "+idx_begin_currAuth_in_eachLine+ " "+idx_begin_from_colon +" "+(idx_begin_currAuth_in_eachLine - idx_begin_from_colon)
//													+" "+idx_begin_currAuth+" eachLine:"+eachLine+"\n"
//													);
						
					}
					writer_dirtyAuthNAme.close();
					
					
					
					return true;
				}
				
			}
		
		}
		catch(Exception e){
			e.printStackTrace();
		}
		return false;
	}
	
	// conv_2_mallet_format  MALLET
    public static boolean conv_2_mallet_format(
    										String  baseFolder,
    										String 	inputFile,
    										String 	delimiter_for_inputFile,
    										String  inFile_organiz_person_locations_numbers_docID,
    										String 	delimiter_for_inFile_organiz_person_locations_numbers_docID,
    										int 	index_of_token_having_bodyText_inputFile,
    										int 	index_of_token_having_title_inputFile,
    										int		index_of_token_having_keywords,
    										int		index_of_token_having_docID,
    										int		index_of_token_having_primarykey,
    										int		index_of_token_having_authName,
    										int 	index_of_token_having_sourceFileName,
    										int		index_of_token_containing_missingKEYWORDS,
    										int	  	index_of_token_containing_bool_POLIorTRAD,
    										String  inFile10_dirty_authNames,
    										String 	outFile,
    										String 	outFile_removed_DIRTY_authName_remDupBYurl,
    										String 	outFile_for_PLSI_input_only_bodyText,
    										boolean run_for_only_inFile10_dirty_authNames,
    										String replace_string_for_sourceChannelFilePath
    										){
    	String inFile11_all_txt_from_inFil10="";
    	TreeMap<Integer,String> map_docID_from_inFile11_as_KEY=new TreeMap<Integer, String>();
    	
    	BufferedReader br=null;
    	String keywords="";
    	// lineNo is the DocID
    	TreeMap<String, String> map_lineNo_number=new TreeMap<String, String>();
    	TreeMap<String, String> map_lineNo_location=new TreeMap<String, String>();
    	TreeMap<String, String> map_lineNo_person=new TreeMap<String, String>();
    	TreeMap<String, String> map_lineNo_organization=new TreeMap<String, String>();
    	TreeMap<String, String> map_lineNo_PRIMARY_key=new TreeMap<String, String>();
    	
    	/// ***********************************************************/
    	//this line no is not DOCID ..we have separate docID
    	TreeMap<String, String> map_lineNo_title=new TreeMap<String, String>();
    	TreeMap<String, String> map_lineNo_keywords=new TreeMap<String, String>();
    	TreeMap<String, String> map_lineNo_DocID=new TreeMap<String, String>();
    	TreeMap<String, String> map_lineNo_primaryKey=new TreeMap<String, String>();
    	TreeMap<String, Integer> map_primaryKey_docID_file1=new TreeMap<String, Integer>();
    	int file1_max_docID=-1;int file2_max_docID=-1;int file3_max_docID=-1;
    	String folder_dirtyAuthorName=baseFolder+"dirtyAuthorName/";
    	TreeMap<Integer,String> map_eachLine_inputFile=new TreeMap<Integer, String>();
    	String 	authName="";FileWriter writerdebug=null;
	try{
		writerdebug=new FileWriter(new File(baseFolder+"debug_cleaning_and_convert.txt"));
		
		TreeMap<Integer,String> map_eachLine_dirtyAuthName=
				  readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
							 .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
									 												 inFile10_dirty_authNames, 
																				 	 -1, //startline, 
																					 -1, //endline,
																					 " reloading..", //debug_label
																					 false //isPrintSOP
																					 );

		// run_for_only_inFile10_dirty_authNames
		if(run_for_only_inFile10_dirty_authNames){
			
			// 
			if( ! (new File(folder_dirtyAuthorName).exists())  ){
				new File(folder_dirtyAuthorName).mkdir();
			}
			else{ //DELETE all files in existing 
				String [] arr_exist_files=new File(folder_dirtyAuthorName).list();
				int c=0;
				while(c<arr_exist_files.length){
					new File(arr_exist_files[c]).delete();
					System.out.println("DELETED file:"+arr_exist_files[c]);
					c++;
				}
			}
			// 
			map_eachLine_inputFile=
						  readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
						 .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
								 												 inputFile, 
																			 	 -1, //startline, 
																				 -1, //endline,
																				 " inputFile..", //debug_label
																				 false //isPrintSOP
																				 );

//			for(int seq2_:map_eachLine_dirtyAuthName.keySet()){
//				System.out.println("----------------------------------------------------------------");
//				String curr_dirty_authName=map_eachLine_dirtyAuthName.get(seq2_);
//				String outFile_for_curr_dirtyAuthName=folder_dirtyAuthorName+curr_dirty_authName.replace(" ", "") +".txt";
//				System.out.println("curr_dirty_authName:"+curr_dirty_authName);
//				System.out.println("outFile_for_curr_dirtyAuthName:"+outFile_for_curr_dirtyAuthName);
//				FileWriter writer_dirtyAuthNAme=new FileWriter(outFile_for_curr_dirtyAuthName);
//				int count_currDirtyAuthName=0;
				String[] s=null;
				// 
				for(int seq_:map_eachLine_inputFile.keySet()){
//					System.out.println("seq_(inputFile):"+seq_);
					String orig_eachLine=map_eachLine_inputFile.get(seq_);
					String eachLine=orig_eachLine.toLowerCase();
					
					s =eachLine.split(delimiter_for_inputFile);
					// 
					authName=s[index_of_token_having_authName -1].toLowerCase();
					authName=authName.replace("from: ", "");
					
//					System.out.println("authName:"+authName);
					// is_given_authName_a_dirtyOne
					boolean is_dirty=is_given_authName_a_dirtyOne(	authName, 
																	map_eachLine_dirtyAuthName,
																	folder_dirtyAuthorName,
																	orig_eachLine, //writes this if only next param is true
																	true //is_write_to_outputFile_dirtyAuthName
												 				 );
					
//					if(is_dirty==true){
//						writer_dirtyAuthNAme.append(orig_eachLine+"\n");
//						writer_dirtyAuthNAme.flush();
//						count_currDirtyAuthName++;
//					}
					
				}
//				System.out.println("count_currDirtyAuthName:"+count_currDirtyAuthName+" s.len:"+s.length );
//				writer_dirtyAuthNAme.close();
//			}
			
				 
				// readFile_eachLine_get_a_particular_token_calc_frequency
				readFile_eachLine_get_a_particular_token_calc_frequency.
				calc_Frequency_Of_Token_AND_output_unique_values_in_token(
																			inputFile, 
																			"!!!", //delimiter_4_inputFolderName_OR_Single_File, 
																			inputFile+"_removedDirtyAuthNames_debug.txt",  //outputFileName1 
																			false,  //is_Append_outputFileName1
																			inputFile+"_removedDirtyAuthNames_onlyKEYWORDS_Debug.txt", //outputFileName_onlykeywords
																			false, //is_Append_outputFileName_onlykeywords, 
																			inputFile+"_removedDirtyAuthNames__WORD_DOCUMENT_FREQ_debug.txt", //outputFileName_WORD_DOCUMENT_FREQ, 
																			index_of_token_having_authName, //index_of_token_interested_for_freq_count, 
																			index_of_token_having_primarykey,// index_of_token_of_SourceURL, 
																			index_of_token_having_sourceFileName,// index_of_token_of_Source_FileName,
																			replace_string_for_sourceChannelFilePath,
																			true, //is_token_interested__authors, 
																			false, //is_split_on_blank_space_in_token, 
																			false, //is_do_stemming_on_word, 
																			"",//YES_Filter, 
																			"",//NO_Filter,
																			true, //isMac,
																			true, //is_split_CSV_on_the_given_token, 
																			baseFolder+"debug_freq_count_authNames.txt",// output_debugFile, 
																			false, //is_add_last_token_of_inputFolderName_to_OUTPUT, 
																			false //isSOPprint
																			);
				
				   
			 
				
			System.out.println("ONLY run_for_only_inFile10_dirty_authNames DONE!!!");
			return true;
		}
		else if(run_for_only_inFile10_dirty_authNames==false){
			inFile11_all_txt_from_inFil10=folder_dirtyAuthorName+"all.txt";
			
			TreeMap<Integer,String> map_eachLine_inFile11_all=
					  readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
								 .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
										 												 inFile11_all_txt_from_inFil10, 
																					 	 -1, //startline, 
																						 -1, //endline,
																						 " reloading..", //debug_label
																						 false //isPrintSOP
																						 );
			// get docID from all.txt ..this docID has to be NO FILTERED (remove docs with dirty author names)
			for(int seq3_:map_eachLine_inFile11_all.keySet()){
				String eachLine_inFile11=map_eachLine_inFile11_all.get(seq3_);
				String []s=eachLine_inFile11.split("!!!");
                //NO FILTER doc_id for dirty author names
				map_docID_from_inFile11_as_KEY.put(Integer.valueOf(s[index_of_token_having_docID-1]), "-1");
			}
		}

 		// map_inFile_organiz_person_locations_numbers_docID
		TreeMap<Integer, String> map_inFile_organiz_person_locations_numbers_docID =
				readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
				readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
																			inFile_organiz_person_locations_numbers_docID,
													    				   -1,
													    				   -1,
													    				   "", //debug_label,
													    					false //isPrintSOP,
													    		           );
		// get numbers, location + person ,  
		for(int seq:map_inFile_organiz_person_locations_numbers_docID.keySet()){
			if(seq==1){continue;} // header skip
			
			String [] s =map_inFile_organiz_person_locations_numbers_docID.get(seq).split(delimiter_for_inFile_organiz_person_locations_numbers_docID);
			
			String person =s[1]; //person 
			String organization =s[2]; //organization
			String location =s[3]; //location
			String numbers=s[4]; //number
			int docID= Integer.valueOf( s[5]);
			String url_primaryKey1=s[6];
			
			String  checksum_primary_key2= clean_retain_only_alpha_numeric_characters.clean_retain_only_alpha_numeric_characters(url_primaryKey1, true);
										// calc_checksum.calc_checksum_for_string(url_primaryKey1);
			
			//  -1 as we have header 
			map_lineNo_number.put(checksum_primary_key2, person);
			map_lineNo_organization.put(checksum_primary_key2, organization);
			map_lineNo_location.put(checksum_primary_key2, location);
			map_lineNo_person.put(checksum_primary_key2, numbers);
			
			//debug
//			if(url_primaryKey1.equals("http://www.thenews.com.pk/latest/116708-sadiq-khan-son-of-a-bus-driver-set-for-londons-mayor-office")
//				|| checksum_primary_key2.equalsIgnoreCase("0bba5d05b982028cb608adaadd2a2347eb77810c")){
				writerdebug.append("----------------------\n");
				writerdebug.append("\ndocID:"+docID+" url:"+url_primaryKey1+" checksum_primary_key2:"+checksum_primary_key2
									+" put.organization:"+organization.length()
									+" put.numbers:"+numbers.length()+" put.location"+location.length()+"<---"+"\n\n");
				writerdebug.append("----------------------\n");
				writerdebug.flush();
//			}
			 
			file1_max_docID=docID;
		}
		
		br = new BufferedReader(new FileReader(inputFile));
		
		String outFile_after_TR=outFile+"_pre_TR.txt";
		String outFile_after_TR2=outFile+"_pre_TR2.txt";
		
		FileWriter writer= new FileWriter(new File(outFile_after_TR));
		FileWriter writer_removed_DIRTY_authName_remDupBYurl=new FileWriter(new File(outFile_removed_DIRTY_authName_remDupBYurl));
		FileWriter writer_outFile_for_PLSI_input_only_bodyText= new FileWriter(new File(outFile_for_PLSI_input_only_bodyText));
		
		TreeMap<String, Integer> map_isAlready_existsSOURCEurl_as_KEY=new TreeMap<String, Integer>();
		
		String line=""; int lineNumber=0; String nline="";
		TreeMap<String, String> map_StopWords=stopwords.stopwords();
        TreeMap<String, String> map_lineNo_bool_TRADorPOLI=new TreeMap<String, String>();
        TreeMap<String, String> map_lineNo_missingKeywords=new TreeMap<String, String>();
		
		while( (line = br.readLine()) !=null){
                            lineNumber++;
                            String orig_eacLine=line;
                            System.out.println("lineNumber:"+lineNumber);
                            line=line.toLowerCase();
                            
                            String []s =line.split(delimiter_for_inputFile);
                            
                            String 	bodyText=s[index_of_token_having_bodyText_inputFile-1];
                            String 	title=s[index_of_token_having_title_inputFile-1];
                              		keywords=s[index_of_token_having_keywords-1];
                            String 	primary_key1=s[index_of_token_having_primarykey-1];
                            		authName=s[index_of_token_having_authName -1];
                            		
                            String  checksum_primary_key1=clean_retain_only_alpha_numeric_characters.
                            							  clean_retain_only_alpha_numeric_characters(primary_key1, true);
//                            		calc_checksum.calc_checksum_for_string(primary_key1);
                            		
                            int  	docID= Integer.valueOf(s[index_of_token_having_docID-1]);
                            String  bool_TRAD_or_POLI= s[index_of_token_containing_bool_POLIorTRAD-1];
                            String  missingKeywords=s[index_of_token_containing_missingKEYWORDS-1];
                            
                            //NO FILTER doc_id for DIRTY author names
                            if(  map_isAlready_existsSOURCEurl_as_KEY.containsKey(checksum_primary_key1)
                            	){
                            	
                            	if(map_isAlready_existsSOURCEurl_as_KEY.containsKey(checksum_primary_key1)){
	                            	writerdebug.append("already exists URL (duplicate): url:"+checksum_primary_key1   +" curr docID:"+docID
	                            						+" already URL with docID:"+map_isAlready_existsSOURCEurl_as_KEY.get(checksum_primary_key1)
	                            						+"\n");
	                            	writerdebug.flush();
                            	}
                            	
                            	continue;
                            }
                            else{
                            	// Docs without DIRTY authName write to a file
                            	writer_removed_DIRTY_authName_remDupBYurl.append(orig_eacLine+"\n");
                            	writer_removed_DIRTY_authName_remDupBYurl.flush();
                            	map_isAlready_existsSOURCEurl_as_KEY.put(checksum_primary_key1, docID); // soureURL
                            }
                            
                            // skip DIRTY authName 
            				// is_given_authName_a_dirtyOne
//            				boolean is_dirty=is_given_authName_a_dirtyOne(	authName, 
//            																map_eachLine_dirtyAuthName,
//            																folder_dirtyAuthorName,
//            																orig_eacLine, //writes this if only next param is true
//        																	false // 
//            											 				 );
//            				if(is_dirty==true){
//            					continue;
//            				}
                            
                            if(title.indexOf("keywords:")>=0){
                            	title= title.substring(0, title.indexOf("keywords:"));
                            }
                            title=title.replace("subject:", "");
                            //remove , in keywords
                            keywords=keywords.replace(",", " ").replace("  ", " ");
                            //TITLE
                            map_lineNo_title.put(checksum_primary_key1, title);
                            //keywords
                            map_lineNo_keywords.put(checksum_primary_key1, keywords);
                            
                            map_lineNo_primaryKey.put(checksum_primary_key1, primary_key1);
//                            map_lineNo_DocID.put(docID, docID);
                            
                            map_primaryKey_docID_file1.put(checksum_primary_key1, docID);
                            
                            map_lineNo_bool_TRADorPOLI.put(checksum_primary_key1, bool_TRAD_or_POLI);
                            map_lineNo_missingKeywords.put(checksum_primary_key1, missingKeywords);
                            
                            
                            bodyText=bodyText.replace("&/039", " ").replace("&039", " ")
			                                 .replace("&lt", "").replace("&gt", "")
			                                 .replace(";;", " ").replace("&/160", "")
			                                 .replace("bt:", "").replace("/”", "")
			                                 .replace(":", " ").replace(",", " ")
			                                 .replace(";", " ")
			                                 .replace("bt:body:: ", "").replace(";br;", "")
			                                 .replace("bt:body::", "").replace("  ", " ")
			                                 .replace("  ", " ");
                            
                            bodyText=removeUnicodeChar.removeUnicodeChar(bodyText);
            
                            if(line.length()>=1){
                				//writer.append( line.substring(1, line.length()) +"\n");
                            	
                            	//ONLY 1 interested token
                				writer.append(  
//                								docID
                								checksum_primary_key1
                								+" "+
                								bodyText.replace("\t", " ")
                                                +"\n"
                                              );
                				writer.flush();
                			}
                            else{
                            	writer.append(  
//                            					docID
                            					checksum_primary_key1
        										+" "
        										+"dummy"
                                        		+"\n"
                                      );
                            	writer.flush();
                            }
                            file3_max_docID=docID; 
		
		}
		
			//	read file after tr -dc [:alnum:][\ ,.]\\n < inputfile > outputfile
			
			// open up standard input
			BufferedReader br2 = new BufferedReader(new InputStreamReader(System.in));
			String continue_flag="N";
			 
				// prompt the user to enter their OPTION
				try {
					System.out.println("converting outFile->"+outFile +" with tr -dc");
					System.out.println("RUNNING... tr -dc [:alnum:][\\ ,.]\\\\n < inputfile > outputfile ");
					System.out.println("RUN BELOW...");
					System.out.println("tr -dc [:alnum:][\\ ,.]\\\\n < "+ outFile_after_TR +" > "+outFile_after_TR2 );
					System.out.println("---------------------------------");
					System.out.println("After running above, Press y to continue .. ");
					
					continue_flag = br2.readLine();
					
					
				} catch (IOException ioe) {
					System.out.println("IO error trying to read your option!");
					System.exit(1);
				}
//				int doc_id=0;
				///
				if(continue_flag.equalsIgnoreCase("y")){
					
			    	BufferedReader br3=null;
			    	 
			    		br3 = new BufferedReader(new FileReader(outFile_after_TR2));
					
			    		writer= new FileWriter(new File(outFile));
			    		
			    		line="";   lineNumber=0;   nline="";
			    		////// 
			    		while( (line = br3.readLine()) !=null){
			    			lineNumber++;
			    			 
			    			
			    			String firstLine="";
			    			if(line.indexOf(".")>=0)
			    				firstLine=line.substring(0, line.indexOf("."));
			    			else if(line.length()>200)
			    				firstLine=line.substring(0, 200);
			    			else if(line.length()>100)
			    				firstLine=line.substring(0, 100);
			    			else if(line.length()>50)
			    				firstLine=line.substring(0, 50);
			    			else
			    				firstLine=line;
			    			
			    			String []s =line.split(" ");
			    			 
			    			String curr_primary_key_CHECKSUM=  s[0]  ;  //URL (primary key)
			    			
			    			if(lineNumber==1){
			    				System.out.println("curr_primary_key:"+curr_primary_key_CHECKSUM+" line:"+line);
			    			}
			    			
			    			String curr_numbers=convert_mapValue_get_only_values_as_String(map_lineNo_number.get(curr_primary_key_CHECKSUM));
			    			String curr_organization=convert_mapValue_get_only_values_as_String(map_lineNo_organization.get(curr_primary_key_CHECKSUM));
			    			String curr_person=convert_mapValue_get_only_values_as_String(map_lineNo_person.get(curr_primary_key_CHECKSUM));
			    			String curr_location=convert_mapValue_get_only_values_as_String(map_lineNo_location.get(curr_primary_key_CHECKSUM));
			    			String curr_title=map_lineNo_title.get(curr_primary_key_CHECKSUM);
			    			String curr_primary_key=map_lineNo_primaryKey.get(curr_primary_key_CHECKSUM);
			    			String curr_keywords="";
//			    			if(map_lineNo_keywords.containsKey(curr_keywords))
			    				curr_keywords=map_lineNo_keywords.get(curr_primary_key_CHECKSUM).replace("keywords:", "");
//			    			else 
//			    				curr_keywords="dummy";
				    		
			    			String curr_bool_TRADorPOLI="";
			    			
//			    			if(map_lineNo_bool_TRADorPOLI.containsKey(curr_primary_key_CHECKSUM))
			    					curr_bool_TRADorPOLI=  map_lineNo_bool_TRADorPOLI.get(curr_primary_key_CHECKSUM).replace("1", "poli").replace("0", "tran");
//			    			else{
//			    				curr_bool_TRADorPOLI= "dummy";
//			    				System.out.println("curr_bool_TRADorPOLI: not found for:"+curr_primary_key_CHECKSUM );
//			    			}
			    			
			    					
			    		    String curr_missingKeywords= map_lineNo_missingKeywords.get(curr_primary_key_CHECKSUM);
			    			
//			    		    System.out.println("curr_title:"+curr_title);
			    			//convert_fn_remove_stopwords
			    			curr_title=convert_fn_remove_stopwords( map_StopWords, curr_title ).replace("'", "");
			    			
			    			writerdebug.append(curr_primary_key_CHECKSUM+" url:"+curr_primary_key+"\n");
			    			writerdebug.flush();
			    			
			    			//debug
			    			if(curr_primary_key.equals("http://www.thenews.com.pk/latest/116708-sadiq-khan-son-of-a-bus-driver-set-for-londons-mayor-office")){
			    				writerdebug.append("curr_numbers:"+curr_numbers+" curr_organization:"+curr_organization+" curr_person:"+curr_person
			    									+" curr_location:"+curr_location +" curr_title:"+curr_title +" curr_keywords:"+curr_keywords
			    									+"\n");
			    				writerdebug.append("map_lineNo_organization:"+map_lineNo_organization+"\n"
//			    								  +" map_lineNo_person:"+map_lineNo_person
//			    								  +" map_lineNo_number:"+map_lineNo_number
			    								  +" map_lineNo_organization.get(curr_primary_key):"+map_lineNo_organization.get(curr_primary_key) +" for lineNumber:"+lineNumber
			    								  +" curr_primary_key:"+curr_primary_key
			    								  +" curr_primary_key_CHECKSUM:"+curr_primary_key_CHECKSUM+"\n"
			    								  );
			    				writerdebug.flush();
			    			}
			    			
			    			//cleaning
			    			curr_numbers=curr_numbers.replace("'", " ").replace("\"", " ").replace("  ", " ");
			    			curr_organization=curr_organization.replace("'", " ").replace("\"", " ").replace("  ", " ");
			    			curr_person=curr_person.replace("'", " ").replace("\"", " ").replace("  ", " ");
			    			curr_location=curr_location.replace("'", " ").replace("\"", " ").replace("  ", " ");
			    			curr_title=curr_title.replace("'", " ").replace("\"", " ").replace("  ", " ");
			    			curr_keywords=curr_keywords.replace("'", " ").replace("\"", " ").replace("  ", " ");
			    			curr_primary_key=curr_primary_key.replace("'", " ").replace("\"", " ").replace("  ", " ");
			    			firstLine=firstLine.replace("'", " ").replace("\"", " ").replace("  ", " ");
			    			//first token is unique primaryKEY , hence remove it.
			    			if(firstLine.indexOf(" ")>0)
			    				firstLine=firstLine.substring(firstLine.indexOf(" "), firstLine.length());
			    			
			    			
			    			line=line.replace("'", " ").replace("\"", " ").replace("  ", " ");
			    			
			    			//first column value is docID , there is a space after that..we need to start from this space to end of line to get bodyText
			    			int begin_idx= line.indexOf(" ");
			    			
			    			line=line.substring(begin_idx+1, line.length());
			    			
			    			System.out.println("doc_id:"+lineNumber);
//			    								+" line:"+line+" begin_idx:"+begin_idx);
			    			
			    			//add double quotes
			    			curr_primary_key="\""+curr_primary_key.replace("!", " ") +"\"";
			    			curr_numbers="\""+" numbers "+curr_numbers+"\"";
			    			curr_organization="\""+" organization "+curr_organization+"\"";
			    			curr_person="\""+ " person location "+curr_person;
			    			curr_location=  curr_location+"\"";
			    			curr_title="\""+" title "+curr_title+"\"";
			    			curr_keywords="\""+" keywords "+ curr_keywords+"\"";
			    			firstLine="\" firstline "+firstLine+"\"";
			    			line="\""+line+"\"";
			    			curr_bool_TRADorPOLI="\""+curr_bool_TRADorPOLI+"\"";
			    			curr_missingKeywords="\""+curr_missingKeywords+"\"";
			    			
			    			// writing writing <docID\tlabel\tbodyText\tnumbers\tperson+location\torganization\tfirstline\ttitle\tkeywords>
			    			if(line.length()>=1){
								//writer.append( line.substring(1, line.length()) +"\n");
			    				
			    				String all_features_together= curr_numbers+" "+curr_person+" "+curr_location
																+" "+curr_organization
																+" "+firstLine
																+" "+curr_title
																+" "+curr_keywords
																+" "+curr_missingKeywords;
//																+" "+curr_bool_TRADorPOLI;
			    				all_features_together="\""+all_features_together.replace("\"", "")+"\"";
			    				
			    				// LDA (MALLET)
								writer.append(   "A-"+lineNumber //DOCID
												+"\t"+"Clabl"
												+"\t"+line.replace("\t", " ")
												+"\t"+curr_numbers
												+"\t"+curr_person+" location2 "+curr_location
												+"\t"+curr_organization
												+"\t"+firstLine
												+"\t"+curr_title
												+"\t"+curr_keywords
												+"\t"+curr_missingKeywords
												+"\t"+curr_bool_TRADorPOLI
												+"\t"+all_features_together
												+"\t"+ curr_keywords.substring(0,curr_keywords.length()-1)
													 +" "+curr_missingKeywords.substring(1, curr_missingKeywords.length()) // (applied tags and missing related tags together)
												+"\t"+curr_primary_key
				                                +"\n"
				                              );
								writer.flush();
								//PLSI
								writer_outFile_for_PLSI_input_only_bodyText.append(line.replace("\t", " ")+"\n");
								writer_outFile_for_PLSI_input_only_bodyText.flush();
								
								file2_max_docID= lineNumber; // doc_id;
							}
			    			else{
			     				String all_features_together= curr_numbers+" "+curr_person+" "+curr_location
																+" "+curr_organization
																+" "+firstLine
																+" "+curr_title
																+" "+curr_keywords
																+" "+curr_missingKeywords;
//																+" "+curr_bool_TRADorPOLI;
			     				all_features_together= "\""+ all_features_together.replace("\"", "") +"\"";
			    				
			    				// LDA (MALLET)
			    				writer.append(   "A-"+lineNumber //DOCID
												+"\t"+"Clabl"
												+"\t"+"\"dummy\"" //bodyTEXT
												+"\t"+curr_numbers
												+"\t"+curr_person+" location2 "+curr_location
												+"\t"+curr_organization
												+"\t"+firstLine
												+"\t"+curr_title
												+"\t"+curr_keywords
												+"\t"+curr_missingKeywords
												+"\t"+curr_bool_TRADorPOLI
												+"\t"+all_features_together
												+"\t"+ curr_keywords.substring(0,curr_keywords.length()-1)
													 +" "+curr_missingKeywords.substring(1, curr_missingKeywords.length()) // (applied tags and missing related tags together)
												+"\t"+curr_primary_key
				                                +"\n"
			    							);
			    				writer.flush();
			    				//PLSI
			    				writer_outFile_for_PLSI_input_only_bodyText.append("dummy"+"\n");
								writer_outFile_for_PLSI_input_only_bodyText.flush();
			    				
			    				file2_max_docID=lineNumber; // doc_id;
			    			}
			    		}
			    		 
				}
			 
				System.out.println("size of maps: map_lineNo_person.size:"+map_lineNo_person.size()
										+" map_lineNo_location:"+map_lineNo_location.size()
										+" map_lineNo_organization:"+map_lineNo_organization.size()
										+" map_lineNo_number:"+ map_lineNo_number.size()
										+" map_lineNo_title.size:"+map_lineNo_title.size()
										+" map_inFile_organiz_person_locations_numbers_docID:"+map_inFile_organiz_person_locations_numbers_docID.size()
								);
				
				System.out.println("file1_max_docID:"+file1_max_docID+" file2_max_docID:"+file2_max_docID
									+" file3_max_docID:"+file3_max_docID);
				//
				if(file1_max_docID!=file2_max_docID || file1_max_docID != file3_max_docID ){
					System.out.println("ERROR: mismatch in 2 input files..NO DOCUMENTS (lines) should be SAME!!!"+file1_max_docID
														+" <-> "+file2_max_docID+" <-> "+file3_max_docID);
				}
				
            }
            catch(Exception e){
                System.out.println("error:"+e.getMessage());
                e.printStackTrace();
            }
		return true;
    }
    
	//(1) pick a token (2) english_only
	public static TreeMap<Integer,Integer> t2_english_only( 
										String 	inputFile, //INPUT
										String  out_repository_file,//output if is_run_for_global_repository==true,otherwise NOT USED AT ALL..
										String  past_alreadyRan_repository_file, // INPUT (if is_run_for_global_repository==FALSE,THIS ONE IS USED to grab if a URL is english or NOT)
										String 	outFile_only_int_token,
										String  outFile_all,
										int 	index_interested_token, //bodyText or Title
										int 	index_containing_primary_key,
										boolean	is_run_for_global_repository,
										String 	debugFile,
										boolean isSOPprint
										){
		TreeMap<Integer,Integer> mapOutStat=new TreeMap<Integer, Integer>();
		BufferedReader br=null;
		FileWriter writer_repository = null;
		FileWriter writer= null;
		FileWriter writer_all=null;
		FileWriter writer_all_NOT_ENGLISH=null;
		int total_matched_english_count=0;
		try{
			System.out.println("out_repository_file:"+out_repository_file);
	        // is_run_for_global_repository
			if(is_run_for_global_repository){
				writer_repository=new FileWriter(new File(out_repository_file), true ); //ALWAYS APPEND ONLY
			}
			else{
				writer_repository=new FileWriter(new File(past_alreadyRan_repository_file), true ); //ALWAYS APPEND ONLY
			}
				
				//REPOSITORY FROM PAST
				TreeMap<String,String> map_primaryKey_languageDetected_REPOSITORY=
		 				LoadMultipleValueToMap2_AS_KEY_VALUE.LoadMultipleValueToMap2_AS_KEY_VALUE(
		 																							past_alreadyRan_repository_file,
		 																							"!!!",
		 																							"1,2",
		 																							"dummy", //append_suffix_at_each_line
		 																							false,   //is_have_only_alphanumeric
		 																							false);
				
						
			br = new BufferedReader(new FileReader(inputFile));
			//NOT FOR REPOSITORY ONLY FILE WILL BE CREATED
			if(is_run_for_global_repository==false){
			  writer= new FileWriter(new File(outFile_only_int_token));
			  writer_all= new FileWriter(new File(outFile_all));
			  writer_all_NOT_ENGLISH= new FileWriter(new File(outFile_all+"__NOT_ENGLISH.txt"));
			}
			FileWriter writerDebug= new FileWriter(new File(debugFile));
			String line=""; int lineNumber=0; String nline="";
			String temp=""; int total_no_doc=0; int english_only_doc=0;
			String curr_primaryKey="";
			while( (line = br.readLine()) !=null){
				String originalLine=line;
				line=line.toLowerCase();
                lineNumber++;
                System.out.println("lineNumber:"+lineNumber);
				if(line.length()>=1){
							total_no_doc++;
                            String keyword="";
                            String []s=originalLine.split("!!!");
 
                            if(s.length<index_interested_token){
                            	writerDebug.append("\nNOT sufficient tokens:total_no_doc:"+total_no_doc+
                            				" line:"+line);
                            	writerDebug.flush();
                            	continue;
                            }
                            
                            temp=s[index_interested_token-1].replace("bt: " , "").replace("bt:body:: ", "")
                            						.replace("bt:" , "").replace("bt:body::", "");
                            
                            curr_primaryKey=s[index_containing_primary_key-1]; 
                            
                            String is_eng2=null;
                            if(map_primaryKey_languageDetected_REPOSITORY.containsKey(curr_primaryKey)){
                            	//already in repository and IF FLAG run only repository , so CONTINUE..NO NEED REWRITE
                            	if(is_run_for_global_repository){ continue;} 
                            	
                            	is_eng2=map_primaryKey_languageDetected_REPOSITORY.get(curr_primaryKey);
                            	
                            }
                            else{
                             	  // find_Language_from_String
	                              is_eng2=find_Language_from_String.find_Language_from_String(
	                            															temp,
	                            															false //isSOPprint
	                            															);
	                               
		  		            		//REPOSITORY
		                      	   writer_repository.append(curr_primaryKey+"!!!"+is_eng2+"\n");
		                      	   writer_repository.flush();
	                              
                            }
                            //NULL  just continue; 
  		                    if(is_eng2==null) continue;
                            
                            if(lineNumber<150)
                            	System.out.println("is_eng2:"+is_eng2+" interestd token:"+temp);            
		                    
							//writer.append( line.substring(1, line.length()) +"\n");
		                    if( is_eng2.equalsIgnoreCase("en")){
		                    	english_only_doc++;
		                    	
		                    	//REPOSITORY RUN
		                    	if(is_run_for_global_repository){
		                    
		                    		
		                    	}
		                    	else{ // NOT REPOSITORY RUN.. , OTHERWIE CONTINUE AS ABOVE if(is_run_for_global_repository){ continue;}
			                    	//only int token
			                    	writer.append("\n"+ temp );
			                    	writer.flush();
			                    	
			                    	// all in line
			                    	writer_all.append("\n"+originalLine);
			                    	writer_all.flush();
		                    	}
		                    	
		                    }
		                    else{

		                    	//REPOSITORY RUN
		                    	if(is_run_for_global_repository){
		                    		
		                    	}
		                    	else{// NOT REPOSITORY RUN.. , OTHERWIE CONTINUE AS ABOVE if(is_run_for_global_repository){ continue;}
			                    	writer_all_NOT_ENGLISH.append(originalLine+"\n");
			                    	writer_all_NOT_ENGLISH.flush();
		                    	}
		                    	
		                    	if(isSOPprint){ // 
		                    		writerDebug.append("\n"+s[index_interested_token-1]);
		                    		writerDebug.flush();
		                    	}
		                    }
				}
				
				if(is_run_for_global_repository==false){
					writer.flush();
				}
			}
			
// NOT ENOUGH
//            boolean is_eng=find_Language_from_String.find_Language_from_String_2( s[interested_token]
//										 						  				, debugFile);
			mapOutStat.put(1, total_no_doc);
			mapOutStat.put(2, english_only_doc);
			 
			System.out.println("english_only_doc:"+english_only_doc+" out of total_no_doc->"+total_no_doc);
		}
		catch(Exception e){
			e.printStackTrace();
		}
		return mapOutStat;
	}
	//
	
    //readFile_and_get_only_numbers_from_eachLine (Line can be header or bodyText)
    public static void readFile_and_get_only_numbers_from_eachLine( String inFile, 
    														  		String outFile
    														  		){
    				BufferedReader br=null;
    				try{
    					br = new BufferedReader(new FileReader(inFile));
    					FileWriter writer= new FileWriter(new File(outFile));
    					String line=""; int lineNumber=0; String nline="";
    					while( (line = br.readLine()) !=null){
    						line=line.toLowerCase();
    		                lineNumber++;
    		                                
//    		                                 if(line.indexOf("afghanistan")==-1 && line.indexOf("china")==-1 &&line.indexOf("iran")==-1
//    		                                &&line.indexOf("saudi arabia")==-1&&line.indexOf("syria")==-1&&line.indexOf("india")==-1 
//    		                                && line.indexOf("bangladesh")==-1 &&line.indexOf("russia")==-1&&line.indexOf("pakistan")==-1
//    		                                &&line.indexOf("egypt")==-1){
//    		                                    continue;
//    		                                }
    		                                // 2,3,7,8
    		                                
    						if(line.length()>=1){
    							line=line.toLowerCase().replace("."," ")
    										.replace("USD"," ").replace("$"," ").replace("€", " ")
    										.replace("£", " ")
    										.replace("  ", " ").replace("  ", " ");
    		    			    String [] arr_=line.split(" ");
    		    			    int len_arr_=arr_.length;
    		    			    int cnt=0; String concNumbers="";
    		    			    
    		    			    //
    		    			    while(cnt<len_arr_){
    		    			    	
    		    			    	if(isInteger.isInteger(arr_[cnt])){
    		    			    		if(concNumbers.length()==0)
    		    			    			concNumbers=arr_[cnt];
    		    			    		else{
    		    			    			concNumbers=concNumbers+"!!!"+arr_[cnt];
    		    			    		}	
    		    			    	}		
    		    			    	cnt++;
    		    			    }
    		                    
    		                    System.out.println("s:"+concNumbers);
    							//writer.append( line.substring(1, line.length()) +"\n");
    							writer.append( concNumbers +"\n");
    							writer.flush();
    						}
    					
    					}
    					
    					System.out.println("ENd..");
    					
    	}
    	catch(Exception e){
    		e.printStackTrace();
    	}
    }
    
    //get and write only required Token
    
    //get_title
    public static void get_title_and_apply_univ_voc_id(
								            String              inFile_1,
								            String              inFile_2_voc_word_AND_voc_id,
								            int                 token_for_title_in_inFile_1,
								            String          	outFile
					            ){
		TreeMap<String, String> map_voc_word_voc_id=new TreeMap<String,String>();
		TreeMap<String, String> map_eachLine_=new TreeMap<String,String>();
		String currLine="";
	  try{
			BufferedReader reader=new BufferedReader(new FileReader(inFile_1));
			BufferedReader reader2=new BufferedReader(new FileReader(inFile_2_voc_word_AND_voc_id));
			FileWriter writer = new FileWriter(new File(outFile));
			// vocabulary 
			map_voc_word_voc_id=
			readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
			readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
			   (inFile_2_voc_word_AND_voc_id,
			   -1,
			   -1,
			   "", //outFile,
				false, //is_Append_outFile,
				false, //is_Write_To_OutputFile,
				"labl",// debug_label,
				-1, // token for  Primary key 
				false //isSOPprint 
	           );
			
			
			//
			map_eachLine_=
			readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
			readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
			   (inFile_1,
			   -1,
			   -1,
			   "", //outFile,
				false, //is_Append_outFile,
				false, //is_Write_To_OutputFile,
				"labl",// debug_label,
				-1, // token for  Primary key
				false //isSOPprint 
	           );
			String currTitle="";
			//
			while((currLine=reader.readLine())!=null){
			    String[] arr_line= currLine.split("!!!");
			    currTitle=arr_line[token_for_title_in_inFile_1];
			   
			    String [] arrTitle=currTitle.split(" ");
			    int len_arrTitle=arrTitle.length;
			    int cnt=0; String concTitle="";
			    
			    //
			    while(cnt<len_arrTitle){
			        if(!stopwords.is_stopword(arrTitle[cnt])
			                    && map_voc_word_voc_id.containsKey(arrTitle[cnt])) {
			           
			            if(concTitle.length() ==0 ){
			                concTitle=map_voc_word_voc_id.get(arrTitle[cnt]);
			            }
			            else{
			                concTitle=concTitle+","+map_voc_word_voc_id.get(arrTitle[cnt]);
			            }
			        }
			    }
			    writer.append("\n"+concTitle);
			    writer.flush();
			} //end each line currLine
		}
		catch(Exception e){
		    e.printStackTrace();
		}
    }
    // 
    public static String remove_custom_Tags(String inStrin){
    	
    	inStrin=inStrin.replace("bt:body::", "").replace("t:head:", "").replace("bt:", "")
    			.replace("auth:from::from:", "").replace("cbtn:", "").replace("message-id:message-id::","")
    			.replace("source:", "").replace("t:", "").replace("body::", "").replace("infile:", "")
    			.replace("keywords:keywords::keywords:", "").replace("keywords::", "")
    			.replace("::", " ")
    			.replace("  ", " ")
    			;
    	return inStrin;
    }
    // remove_commas_etc
    public static String remove_commas_etc(String inStrin){
    	inStrin=inStrin.replace(";"," ").replace(","," ").replace(":"," ")
    			.replace("'"," ").replace("\"", " ").replace("’", " ")
    			.replace("-", " ").replace("”", " ").replace("“", " ")
    			.replace("—", " ").replace("(", " ").replace(")", " ")
    			.replace("?", " ")
    			.replace("  ", " ").replace("  ", " ");
    	return inStrin;
    }
    //convert_fn_remove_stopwords
    public static String convert_fn_remove_stopwords( TreeMap<String, String> map_StopWords ,  String inString){
    	try{
    		 
	    		//iterate each stop word and replace
	    		for(String curr_stopword:map_StopWords.keySet() ){
	    			inString=inString.replace(" "+curr_stopword+" ", " ");
	    		}
	    	
    	}
    	catch(Exception e){
    		
    	}
    	return inString;
    }
    
    //readFile_and_get_only_numbers_from_eachLine (Line can be header or bodyText)
    public static void readFile_and_remove_stop_words_for_eachLine( String  inFile,
    																String  delimiter_of_inFile,
    														  		String  outFile,
    														  		int 	index_token_having_bodyText,
    														  		boolean isSOPprint
    														  		){
    	
    	BufferedReader br=null;

		try{
			br = new BufferedReader(new FileReader(inFile));
			FileWriter writer= new FileWriter(new File(outFile));
			String line=""; int lineNumber=0; String nline="";
			
			// 
			writer.append("\nbodyText!!!url!!!dateTime!!!Authors!!!SourceFile!!!bodyTextOriginal\n");
			writer.flush();
			
			while( (line = br.readLine()) !=null){
				
				//line=line.toLowerCase();
				
                lineNumber++;
                System.out.println("(SW) lineNo:"+lineNumber);
                
                String [] arr_tokens=line.split(delimiter_of_inFile);
                String curr_bodyText=arr_tokens[index_token_having_bodyText-1];
                
				if(line.length()>=1){
					//
					line=line.toLowerCase().replace("."," ")
								.replace("USD"," ").replace("$"," ").replace("€", " ")
								.replace("£", " ")
								.replace("  ", " ").replace("  ", " ");
    			    String [] arr_=curr_bodyText.split(" ");
    			    int len_arr_=arr_.length;
    			    int cnt=0; String concStrings="";
    			    
    			    // split and iterate each word of  bodyText
    			    while(cnt<len_arr_){
    			    	// not STOP WORDS
    			    	if(!stopwords.is_stopword(arr_[cnt].toLowerCase())){
    			    		arr_[cnt]= remove_commas_etc(
    			    					removeUnicodeChar.removeUnicodeChar(
    			    					remove_custom_Tags(arr_[cnt])));
    			    		if(concStrings.length()==0)
    			    			concStrings=arr_[cnt];
    			    		else{
    			    			concStrings=concStrings+" "+arr_[cnt];
    			    		}	
    			    	}		
    			    	cnt++;
    			    }
    			    
    			    int cnt2=0;
    			    String concToken_currLine="";
    			    
    			    TreeMap<String, String> map_StopWords=stopwords.stopwords();
    			    String curr_bodyText_ORIG="";
    			    /// 
    			    while(cnt2<arr_tokens.length){
    			    	
			    		/// BODYTEXT
    			    	if((index_token_having_bodyText-1 )  == cnt2 ){
    			    		curr_bodyText_ORIG=arr_tokens[cnt2];
    			    		// lower case
    			    		concStrings=concStrings.toLowerCase();
    			    		//clean
    			    		concStrings=concStrings.replace(".", " ").replace("   ", " ").replace("  ", " ");
    			    		
    			    		//iterate each stop word and replace
    			    		for(String curr_stopword:map_StopWords.keySet() ){
    			    			concStrings=concStrings.replace(" "+curr_stopword+" ", " ");
    			    		}
    			    		
    			    		if(concToken_currLine.length()==0){
    			    			concToken_currLine=concStrings;
    			    			
    			    		}
    			    		else{
    			    			concToken_currLine=concToken_currLine + delimiter_of_inFile + concStrings ;
    			    		}
   
    			    		
    			    	}
    			    	//NOT BODYTEXT
    			    	else{
    			    		if(concToken_currLine.length()==0){
    			    			concToken_currLine=arr_tokens[cnt2];
    			    		}
    			    		else{
    			    			concToken_currLine=concToken_currLine + delimiter_of_inFile+ arr_tokens[cnt2];
    			    		}
    			    	}
    			    	 
    			    	cnt2++;
    			    }
    			    
                    if(isSOPprint)
                    	System.out.println("s:"+concStrings);
					//writer.append( line.substring(1, line.length()) +"\n");
					writer.append( concToken_currLine +  delimiter_of_inFile+ curr_bodyText_ORIG+  "\n");
					writer.flush();
				}
			
			}
			
			System.out.println("ENd..");
			
		}
		catch(Exception e){
			e.printStackTrace();
		}
    }
    
    // calculate ground truth (add all 3 metrics for GT)
    public static void calc_GTruth_Add_TfIDF_AND_dist_NOUN_count(
    											String tdIDF_inFile1,
    											String count_dist_NOUNS_inFile1
    										    ){
    	try{
    		
    	}
    	catch(Exception e){
    		e.printStackTrace();
    	}
    }
    
    //AWK IS now handling this.. inFile_2 has the tokens without 2 tokens having concatenated without space in between problem.
    public static void match_get_bodyText_for_spaceAdded(
    													 String baseFolder,
    													 String inFile_1,
													     String inFile_2,
													     String output_matcheFile
				){
    	TreeMap<Integer, String > map_inFile1_ =new TreeMap<Integer, String>();
    	TreeMap<Integer, String > map_inFile2_ =new TreeMap<Integer, String>();
    	TreeMap<String , Integer > map_inFile1_key_lineNo =new TreeMap<String, Integer >();
    	TreeMap<String, Integer > map_inFile2_key_lineNo =new TreeMap<String, Integer >();
    	TreeMap<String , Integer > map_inFile1_keyURL_lineNo =new TreeMap<String, Integer >();
    	TreeMap<String, Integer > map_inFile2_keyURL_lineNo =new TreeMap<String, Integer >();
    	
    	TreeMap<String , String > map_inFile1_keyURL_bodyText =new TreeMap<String, String >();
    	TreeMap<String, String > map_inFile2_keyURL_bodyText =new TreeMap<String, String >();
    	
    	TreeMap<Integer, Integer > map_LineNo_matched_between_2Files=new TreeMap<Integer, Integer>();
    	FileWriter writerDebug=null; FileWriter writer=null;
    	try{
    		writerDebug=new FileWriter(new File(baseFolder+"debug_spacematched.txt" ));
    		writer=new FileWriter(new File(output_matcheFile ));
    		//inFile1 -> 2, 6th token used as KEY
    		// 2 token ex: auth:from: yosemite.epa.gov-rnd (authors); 
    		// 6 token ex: message-id:message-id: <http://yosemite.epa.gov/opa/admpress.nsf/0/a18481d677f50b6b85257ee40065cae7@localhost.localdomain
    		// 9 -> bodyText
    		map_inFile1_ =
    				readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
    				readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
														    				   inFile_1,
														    				   -1,
														    				   -1,
														    				   "", //debug_label,
														    					false //isPrintSOP,
														    		           );
    		
    		System.out.println("key map_inFile1_");
    		// 
    		for(int lineNo:map_inFile1_.keySet()){
    			String currLine=map_inFile1_.get(lineNo);
    			if(currLine.indexOf("body::")==-1) continue;
    			String []s=map_inFile1_.get(lineNo).split("!!!");
    			if(s.length < 6) continue;
    			
    			String url=s[5].replace("message-id:message-id::message-id:", "")
							.replace("message-id::message-id:", "");
    			
    			map_inFile1_key_lineNo.put(s[1] +url
    									, lineNo);
    			map_inFile1_keyURL_lineNo.put(url
										, lineNo);  //ONLY URLS
    			
    			
    		}
    		
    		//inFile1 -> 2, 9th token used as KEY
    		// 3 -> bodyText
    		map_inFile2_ =
    				readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
    				readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
														    				   inFile_2,
														    				   -1,
														    				   -1,
														    				   "", //debug_label,
														    					false //isPrintSOP,
														    		           );
    		System.out.println("key map_inFile2_");
    		for(int lineNo:map_inFile2_.keySet()){
    			String currLine=map_inFile2_.get(lineNo);
    			if(currLine.indexOf("body::")==-1) continue;
    			String []s=currLine.split("!!!");
    			if(s.length <9) continue;
    			
    			String url=s[8].replace("message-id:message-id::message-id:", "")
						.replace("message-id::message-id:", "");
    			map_inFile2_key_lineNo.put(s[1] +url
    									, lineNo);
    			map_inFile2_keyURL_lineNo.put(url
    										, lineNo);  //ONLY URLS
    			
    			//map_inFile2_keyURL_bodyText.put(url , s[2] ); //bodytext
    			//System.out.println("inFile_2 lineNo:"+lineNo);
    		}
    		System.out.println("matching");
    		// try to match KEY between 2 files
    		for(String key:map_inFile1_keyURL_lineNo.keySet()){
    			
    			// MATCH
    			if(map_inFile2_keyURL_lineNo.containsKey(key)){
    				map_LineNo_matched_between_2Files.put(map_inFile1_keyURL_lineNo.get(key) , 
    													  map_inFile2_keyURL_lineNo.get(key));
    				
    				writer.append(key+"!!!"+map_inFile1_.get(map_inFile1_keyURL_lineNo.get(key))
    							  	  +"!!!<-->"+map_inFile2_.get(map_inFile2_keyURL_lineNo.get(key))+"\n");
    				writer.flush();
    				
    				
    				
    				
    			}
    			
    		}
    		

			
    		
    		//debug file
    		for(int i:map_LineNo_matched_between_2Files.keySet()){
    			writerDebug.append(i+" "+map_LineNo_matched_between_2Files.get(i));
    			writerDebug.flush();
    		}
    		//debug 
    		for(String key:map_inFile1_key_lineNo.keySet()){
    			writerDebug.append(key +" inFile_1\n" );
    			writerDebug.flush();
    		}
    		//debug 
    		for(String key:map_inFile2_key_lineNo.keySet()){
    			writerDebug.append(key +" inFile_2\n" );
    			writerDebug.flush();
    		}
    		
    		//debug 
    		for(String key:map_inFile1_keyURL_lineNo.keySet()){
    			writerDebug.append(key +" inFile_1(URL)\n" );
    			writerDebug.flush();
    		}
    		//debug 
    		for(String key:map_inFile2_keyURL_lineNo.keySet()){
    			writerDebug.append(key +" inFile_2(URL)\n" );
    			writerDebug.flush();
    		}
    		
    		// 
    		System.out.println("map_LineNo_matched_between_2Files.size: "+map_LineNo_matched_between_2Files.size());
    		System.out.println("map_inFile1_key_lineNo.size:"+map_inFile1_key_lineNo.size());
    		System.out.println("map_inFile2_key_lineNo.size:"+map_inFile2_key_lineNo.size());
    		
    	}
    	catch(Exception e ){
    		e.printStackTrace();
    	}
    }
    
	//applying_missing_keywords_AND_sourcePOLIorTRAD
	private static void applying_missing_keywords_AND_sourcePOLIorTRAD(
																String  baseFolder,
																String 	all_unique_matched_lines_on_allQUERIES, //IN
																String 	all_unique_matched_lines_on_allQUERIES_addMISSINGtopic, //OUT 
																int 	token_containing_keywords, 
																int 	token_containing_sourceFile_TRAD_or_POLI,
																String  q_topic_10_CSV_string
																) {
		TreeMap<Integer,String> map_seq_keywords=new TreeMap<Integer, String>();
		TreeMap<Integer,String> map_seq_sourceFile_TRAD_or_POLI=new TreeMap<Integer, String>();
		TreeMap<Integer,Integer> map_seq_sourceFile_TRAD_or_POLI_BOOLEAN=new TreeMap<Integer, Integer>();
		String []arr_query=q_topic_10_CSV_string.split(",");
		// TODO Auto-generated method stub
		try {
			FileWriter writer=new FileWriter(new File(all_unique_matched_lines_on_allQUERIES_addMISSINGtopic));
			FileWriter writerDebug=new FileWriter(new File(baseFolder+"debug_missing_keywords_AND_sourcePOLIorTRAD.txt"));
			
			TreeMap<Integer,String> map_seq_eachLine=
					  readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
								 .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
										 												 all_unique_matched_lines_on_allQUERIES, 
																					 	 -1, //startline, 
																						 -1, //endline,
																						 " reloading..", //debug_label
																						 false //isPrintSOP
																						 );
			//
			for(int seq:map_seq_eachLine.keySet()){
				String orig_eachLine=map_seq_eachLine.get(seq);
				String [] s=orig_eachLine.split("!!!");
				String curr_sourceFile_TRAD_or_POLI=s[token_containing_sourceFile_TRAD_or_POLI-1];
				map_seq_keywords.put(seq, s[token_containing_keywords-1]);
				map_seq_sourceFile_TRAD_or_POLI.put(seq, curr_sourceFile_TRAD_or_POLI);
				///
				if(curr_sourceFile_TRAD_or_POLI.indexOf("_POLI")>=0)
					map_seq_sourceFile_TRAD_or_POLI_BOOLEAN.put( seq, 1);
				else if(curr_sourceFile_TRAD_or_POLI.indexOf("_TRAD")>=0)
					map_seq_sourceFile_TRAD_or_POLI_BOOLEAN.put( seq, 0);
			}
			
			TreeMap<String,TreeMap<String,String>> map_queryASkeyword_missingKEYWORDSasKEYS=new TreeMap<String, TreeMap<String,String>>();
			//each line can have >1 queries 
			TreeMap<Integer, String> map_seq_querySETfound=new TreeMap<Integer, String>();
			
			// find missing keywords
			for(int seq:map_seq_keywords.keySet()){
				String orig_eachLine=map_seq_eachLine.get(seq);
				String eachLine=orig_eachLine.toLowerCase();
				int c2=0;
				// iterate each query AND see if curr document has it 
				while(c2 < arr_query.length){
					String curr_arr_query =arr_query[c2];
					
					// if line has one of the query from input q_topic_10_CSV_string
					if(eachLine.indexOf(curr_arr_query.toLowerCase()) >=0 ){
					
						String  curr_keyword=map_seq_keywords.get(seq);
								curr_keyword=curr_keyword.replace("Keywords:", "");
								
						String [] arr_curr_keyword_split_on_comma=curr_keyword.split(",");
						int c=0;
						//
						while(c<arr_curr_keyword_split_on_comma.length){
							String curr_keyword_after_split=arr_curr_keyword_split_on_comma[c];
							
							if(curr_keyword_after_split.replace(" ", "").equals("dummy") ){
								c++; continue;
							}
							//
							if(!map_queryASkeyword_missingKEYWORDSasKEYS.containsKey(curr_arr_query)){
								TreeMap<String,String> temp=new TreeMap<String, String>();
								temp.put(curr_keyword_after_split, "");
								map_queryASkeyword_missingKEYWORDSasKEYS.put(curr_arr_query, temp);
							}
							else{
								TreeMap<String,String> temp=map_queryASkeyword_missingKEYWORDSasKEYS.get(curr_arr_query);
								temp.put(curr_keyword_after_split, "");
								map_queryASkeyword_missingKEYWORDSasKEYS.put(curr_arr_query, temp);
							}
							c++;
						}
						///
						if(!map_seq_querySETfound.containsKey(seq))
							map_seq_querySETfound.put(seq, curr_arr_query);
						else{
							String query_CSV=map_seq_querySETfound.get(seq)+","+curr_arr_query;
							map_seq_querySETfound.put(seq, query_CSV);
						}
						
					} // END if(eachLine.indexOf(curr_arr_query.toLowerCase()) >=0 ){
					c2++;
				}
				
			}
			
			/// DEBUGGING
			for(String q:map_queryASkeyword_missingKEYWORDSasKEYS.keySet()){
				writerDebug.append("map_queryASkeyword_missingKEYWORDSasKEYS:"+q+"<-->"+map_queryASkeyword_missingKEYWORDSasKEYS.get(q)
								+"<-->"+map_queryASkeyword_missingKEYWORDSasKEYS.get(q).keySet()
										+"\n\n");
				writerDebug.flush();
				System.out.println("map_queryASkeyword_missingKEYWORDSasKEYS:"+q+ " size:"+map_queryASkeyword_missingKEYWORDSasKEYS.get(q).size());
			}
			writerDebug.append("\n\n map_seq_querySETfound:"+map_seq_querySETfound);
			writerDebug.flush();
			
			System.out.println("map_queryASkeyword_missingKEYWORDSasKEYS.size:"+map_queryASkeyword_missingKEYWORDSasKEYS.size());
			TreeMap<Integer,String> map_seq_foundCorrespondingKeywords=new TreeMap<Integer, String>();
			TreeMap<String, String> map_isAlreayd_seqCorrKEYWORDS=new TreeMap<String, String>();
			//
			for(int seq	:map_seq_eachLine.keySet()){
				map_isAlreayd_seqCorrKEYWORDS=new TreeMap<String, String>();
				String currLine=map_seq_eachLine.get(seq);
				String currLine_lower=currLine.toLowerCase();
				String set_of_queries_found_in_currLine=map_seq_querySETfound.get(seq);
				if(set_of_queries_found_in_currLine==null){
					writer.append("dummy!!!"+map_seq_sourceFile_TRAD_or_POLI_BOOLEAN.get(seq)+"\n");
					writer.flush();
					continue;
				}
				System.out.println(" seq:" +seq + " set_of_queries_found_in_currLine:"+set_of_queries_found_in_currLine);
				String []arr_set_of_queries_found_in_currLine=set_of_queries_found_in_currLine.split(",");
				String conc_currLine_foundAllqueries_correspondingmissingKeywords="";
				int c=0;
				while(c<arr_set_of_queries_found_in_currLine.length){
					if(conc_currLine_foundAllqueries_correspondingmissingKeywords.length()==0){
						conc_currLine_foundAllqueries_correspondingmissingKeywords=
								map_queryASkeyword_missingKEYWORDSasKEYS.get( arr_set_of_queries_found_in_currLine[c]).keySet().toString();
					}
					else{
						conc_currLine_foundAllqueries_correspondingmissingKeywords=conc_currLine_foundAllqueries_correspondingmissingKeywords+" "+
								map_queryASkeyword_missingKEYWORDSasKEYS.get( arr_set_of_queries_found_in_currLine[c]).keySet().toString();
						
					}
					c++;
				}
//				System.out.println("conc_currLine_foundAllqueries_correspondingmissingKeywords:"+conc_currLine_foundAllqueries_correspondingmissingKeywords);
				
				/// split on currLine_foundAllqueries_correspondingmissingKeywords and filter only found keywords..
				String []s2=conc_currLine_foundAllqueries_correspondingmissingKeywords.split(",");
				int c2=0;
				while(c2<s2.length){
					String curr_correspondingmissingKeywords=s2[c2];
					System.out.println("seq:"+seq);
//										+" curr_correspondingmissingKeywords:"+curr_correspondingmissingKeywords);
					if(currLine_lower.indexOf(curr_correspondingmissingKeywords ) >=0){
						
						if(map_isAlreayd_seqCorrKEYWORDS.containsKey(curr_correspondingmissingKeywords.replace(" ", ""))){
							if( !map_seq_foundCorrespondingKeywords.containsKey(seq))
								map_seq_foundCorrespondingKeywords.put(seq, curr_correspondingmissingKeywords );
							else{
								String temp= map_seq_foundCorrespondingKeywords.get(seq)+","+curr_correspondingmissingKeywords;
								map_seq_foundCorrespondingKeywords.put(seq, temp );
							}
						}
						
					}
					
					map_isAlreayd_seqCorrKEYWORDS.put(curr_correspondingmissingKeywords.replace(" ", ""), "");
					c2++;
				}
				//WRITE each line found corresponding KEYWORD to this file..
				writer.append(map_seq_foundCorrespondingKeywords.get(seq)+"!!!"+map_seq_sourceFile_TRAD_or_POLI_BOOLEAN.get(seq)+"\n");
				writer.flush();
			}
			
			//debug
//			for(int seq:map_seq_foundCorrespondingKeywords.keySet()){
//				writerDebug.flush();
//			}
			
			
		} catch (Exception e) {
			e.printStackTrace();
			// TODO: handle exception
		}
	}
    
	//merge_missing_keywordFILE_with_allQueriesFILE
	private static void merge_missing_keywordFILE_with_allQueriesFILE(
									String all_unique_matched_lines_on_allQUERIES,
									String all_unique_matched_lines_on_allQUERIES_addMISSINGtopic_AND_sourceCHANNEL,
									String all_unique_matched_lines_on_allQUERIES_merge_add2MOREfeature) {
		// TODO Auto-generated method stub
		try {
			FileWriter writer=new FileWriter(new File(all_unique_matched_lines_on_allQUERIES_merge_add2MOREfeature));
			TreeMap<Integer,String> map_seq_eachLine_all_unique_matched_lines_on_allQUERIES=
					  readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
								 .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
										 												 all_unique_matched_lines_on_allQUERIES, 
																					 	 -1, //startline, 
																						 -1, //endline,
																						 " reloading..", //debug_label
																						 false //isPrintSOP
																						 );
			
			TreeMap<Integer,String> map_seq_all_unique_matched_lines_on_allQUERIES_addMISSINGtopic_AND_sourceCHANNEL=
					  readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
								 .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
										 												 all_unique_matched_lines_on_allQUERIES_addMISSINGtopic_AND_sourceCHANNEL, 
																					 	 -1, //startline, 
																						 -1, //endline,
																						 " reloading..", //debug_label
																						 false //isPrintSOP
																						 );
			
			///
			for(int seq:map_seq_eachLine_all_unique_matched_lines_on_allQUERIES.keySet()){
				
				if(map_seq_eachLine_all_unique_matched_lines_on_allQUERIES.get(seq).length()>3){ //NO BLANK LINES
					writer.append(map_seq_eachLine_all_unique_matched_lines_on_allQUERIES.get(seq)
								 +"!!!"+map_seq_all_unique_matched_lines_on_allQUERIES_addMISSINGtopic_AND_sourceCHANNEL.get(seq)+"\n");
					writer.flush();
				}
				
			}

			
			
		} catch (Exception e) {
			e.printStackTrace();
			// TODO: handle exception
		}
		
	}
	
	// main
	public static void main(String[] args) throws IOException {
		TreeMap<String,String> mapout= new TreeMap<String, String>();
		//MANUAL CHANGE
		String mainFolder="/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/";
		//MANUAL CHANGE
		String pastRan_repository_file=mainFolder+"repo_url_langDetected2.txt";
				
		String baseFolder="/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/";
		String debugFile=mainFolder+"debug.txt";
		String debugFile_MAIN=mainFolder+"debug_MAIN_cleaning_and_convert.txt";
		FileWriter write_debugFile_MAIN=null;
        String inFile="";
        String outFile_only_int_SetOftoken="" ;//"mergedall-2016-T9-10-Topics-only-bodyText.txt";
        String outFile_allTokens_removeSTOPWORDS="" ;
        String outFile_all="";//
		String outFile_all_tokens="";
		int index_of_token_having_authName=4;
  		int token_containing_body_text_WITH_STOPWORDS=8;
//  		int token_containing_body_text_WITH_STOPWORDS=8;
  		int token_containing_title=6;
//  		int token_containing_auth_Name=4;
  		int token_containing_sourceURL=2;
  		int token_containing_keywords=5;
  		int token_containing_sourceFile_TRAD_or_POLI=7;
  		int token_containing_date=3;
  		
		long t0 = System.nanoTime();
		
		// Methods
		//----------
		// get_title() 
		// t2_english_only
		// get_all_text_features_cleaned ( mallet ) 
		// find_get_a_token_and_write_to_another
		// readFile_and_get_only_numbers_from_eachLine
		// load_AuthID_and_AuthName
		// conv_2_mallet_format ( mallet )
		// ** Prepares MALLET input file for LDA 
		//----------
		
		//***PRE-REQUISTION: 
		// (1) use "run_type.equalsIgnoreCase("prob") && method_to_run.equalsIgnoreCase("2")) 
		//   				 separately for TRAD channel and POLICY channel.
		// This extracts necessary feature (author name, text)
		// keep output of "find_get_a_token_match_on_N_Topics_and_write_to_another" as input to crawler RUN
		// (2) then merge output from both channels to get inFile..+ applied filter of AWK on this file on 10 Topics interested.
		//MAIN - all tokens MERGED FILE from crawler.java RUN ***"run_type.equalsIgnoreCase("prob") && method_to_run.equalsIgnoreCase("2")) 
		//to get this file BELOW
		// (3) For flag=5,  
  	    //	(b) with run_for_only_inFile10_dirty_authNames==true and  ==>then  run "cat * > all.txt" from command prompt in folder "dirtyAuthorName"
  	    // 	(c) with run_for_only_inFile10_dirty_authNames==false
		
        inFile=mainFolder+"mergedall-all-tokens-TRAD-POLI.txt";  //INPUT--NOTE: This file will be created by merging of TRAD and POLI by
	    String q_topic_10_CSV_string="india,syria,boko haram,climate change,election,crime";
	    String q_topic_10_CSV_string_NOT="indiana,dummy,dummy,dummy,dummy,dummy";
	    //removed -> zika, cancer, trafficking, kenya, britain
	    int index_token_having_bodyText=1;
        outFile_only_int_SetOftoken=mainFolder+"mergedall-2016-T9-only-bodyText.txt";
        outFile_all_tokens=baseFolder+"mergedall-2016-T9-10-Topics-all-tokens.txt"; //output
        //outFile_all=mainFolder+"mergedall-2016-T9-all-tokens-ENGLISH.txt";
        
  	    // Flag==44 BEGIN
        int top_N_to_be_matched_for_eachQuery=2000;
        int index_token_to_Verify_eachLine_token=9;
	    // Flag==44 END
        
  	    // **** Flag==5 BEGIN
  	  	String inFile10_dirty_authNames="/Users/lenin/Dropbox/workspace-sts-3.4.0.RELEASE.macmini/crawler/src/p6/dirty_authNames"; //FLAG==5
  	    String replace_string_for_sourceChannelFilePath="infile:/users/lenin/downloads/#problems/p6/merged_/rawxmlfeeds2csv/merged/all_all_body_url_datetime_auth_keywor_subjec_";
  	  	boolean is_run_for_only_inFile10_dirty_authNames=false; //FLAG==5
		//to get below, run around 3 methods in main of find_Verb_AND_Noun_JustBfr_from_TaggedString_inFile N get "output_organiz_person_locations_numbers_docID.txt",
	    String inFile_organiz_person_locations_numbers_docID=baseFolder+"output_organiz_person_locations_numbers_docID.txt";	    //2nd input file (Flag==5 needs this)
  	    // **** Flag==5 END
		 ////////////////////////////////////////////////////////////////////////////////
  	  	  int Flag=4402;
		 ///////////////////////////////////////////////////////////////////////////////////////
          //**************** Flag_for_Task ****************************************************************
  	    // Flag==1 -> find_get_a_token_match_on_N_Topics_and_write_to_another
  	    // Flag==2 -> t2_english_only & remove_NOISE
  	  	//  	   #### you have to run crawler.getCrawler() at this point, to run flag==3
  	    // Flag==3 -> readFile_and_remove_stop_words_for_eachLine()
  	    // Flag==4 -> verify the number of tokens
  	    // Flag==44 -> remove the DIRTY authorNames from 2 files(it will auto produce 20000.txt) ---(no relevant-- PREVIOUSLY done , then run "cat * > 20000.txt" 
  	    // Flag==440 -> applying_missing_keywords_AND_sourcePOLIorTRAD() (TIME:7minutes for 12k documents)
  	    // Flag==4401 -> merging output file of FLAG==440 with the file "20000.txt" (from Flag==44) 
    	// Flag==4402 -> (OBSELETE) Cleaning the CRAWLED documents using GLOBAL PATTERN from each source.. 
  	    // Flag==444 -> producing "inFile_organiz_person_locations_numbers_docID" and "find_Verb_AND_Noun_JustBfr_from_TaggedString_inFile"
  	    // Flag==5   -> conv_2_mallet_format() (RUN twice with "is_run_for_only_inFile10_dirty_authNames=true" and then false) 
  	  	// Flag==999 -> Run all <---
          //**************** Flag_for_Task ****************************************************************


	    try{
	    write_debugFile_MAIN=new FileWriter(new File(debugFile_MAIN));
	    
	    if(Flag==1){
           // get match on 10 topics** only //COMMENT AND UNCOMMENT as required
	      find_get_a_token_match_on_N_Topics_and_write_to_another(
					      									 inFile,
					      									 outFile_all_tokens,
					      									 outFile_only_int_SetOftoken,
					      									 index_token_having_bodyText, // interested_token{9=bodyText, 6=Title}
					      									 q_topic_10_CSV_string,
					      									 q_topic_10_CSV_string_NOT,
					      									 false //isSOPprint
	      									 		 		 );
	    }
	    
          //match two files "inFile" and "inFile_spaceAdded" to match up and get bodyText.
	      inFile=outFile_all_tokens; //MAIN ENGLISH ONLY
	      outFile_all=outFile_all_tokens+"_ONLY_ENGLISH.txt";
	      outFile_only_int_SetOftoken=outFile_only_int_SetOftoken+"_ONLY_ENGLISH.txt";
	      System.out.println("inFile:"+inFile);
	      
	      if(Flag==2){
        // NOTE:**** start with file that has ****only 10 required topics**** contained in it.
        // TIME CONSUMING(run on smaller dataset- 10 intrested topics)  
        // (1) pick a token (2) is english, write to another file //COMMENT AND UNCOMMENT as required
			TreeMap<Integer,Integer> mapOutStat=
											t2_english_only(
															inFile,
															"", //
															pastRan_repository_file,
															outFile_only_int_SetOftoken, // OUTPUT
															outFile_all,				 // OUTPUT
															1, // interested_token{9=bodyText}
															2, //index_containing_primary_key == URL
															false , //is_run_for_global_repository ******
															debugFile,
															false //isSOPprint
															);
			
	      
	      /////////////////////  remove NOISE  having ([ or (^	      
		      remove_NOISE(
			    		  		outFile_only_int_SetOftoken , //INPUT
			    		  		outFile_all //INPUT 
		    		  		);
	       
	      }
	      
	      String only_bodyText_removed_NOISE= outFile_only_int_SetOftoken+"removed_NOISE.txt" ;
	      String allTokens_removed_NOISE= outFile_all +"_removed_NOISE.txt";
	      		 outFile_allTokens_removeSTOPWORDS=  allTokens_removed_NOISE+ "_REMOVE_STOPWORDS.txt";
	   	      
	   	      /////////BEGIN Below for readFile_2_Files_match_by_common_primary_key_and_add_a_tokenFrom_SecondFile_to_FirstFile
	      		String input_crawled_FOLDER=baseFolder+"crawling/";
//	   			String  first_File=allTokens_removed_NOISE;
	   			String first_File=baseFolder+"mergedall-2016-T9-10-Topics-all-tokens.txt_ONLY_ENGLISH.txt_removed_NOISE.txt_REMOVE_STOPWORDS_FINAL.txt_MATCHED.txt_rem_DIRTYauthName_remDUPbyURL_BACKUP.txt",//second_File,
	   			
	   			// Below is manually crawled using getCrawler() in crawler.java .. .option 1 (look Flag=4420 notes for more info)
	   			second_File=input_crawled_FOLDER+"crawling_CLEANED.txt";
//	   			String  second_File_cleaned=crawled_matched_file+"_cleanedURL.txt";

	   			String outFile_Exists=first_File+"_addCrawledEXISTS.txt";
	   			String outFile_Not_Exists=first_File+"_NOexists.txt";
	   			String delimiter_to_be_added_to_first_File="";
	   			//primary key created in same given order
	   			String token_in_first_file_having_primary_Key_CSV="2";
	   			//primary key created in same given order
	   			String token_in_second_file_having_primary_Key_CSV="1";
	   			String token_interested_in_second_File_to_be_appended_to_first_CSV="2";
	   			String token_in_first_file_WHERE_APPENDING_token_interested_in_second_File_to_be_appended_to_first_CSV="1"; // 1= bodyTExt
	   			//	Don't change this is used inside "readFile_2_Files_match_by_common_primary_key_and_add_a_tokenFrom_SecondFile_to_FirstFile"
	   			String output_both=outFile_Exists+"_BOTH.txt" ;
	   			
	   		///////END Below for readFile_2_Files_match_by_common_primary_key_and_add_a_tokenFrom_SecondFile_to_FirstFile
	      /////////
	      if(Flag==3){
	    	  	
	    	   // merge BodyText from RSS Feed and BodyText from CRAWLIN WEB news link ..
				//readFile_2_Files_match_by_common_primary_key_and_add_a_tokenFrom_SecondFile_to_FirstFile
				readFile_2_Files_match_by_common_primary_key_and_add_a_tokenFrom_SecondFile_to_FirstFile.
				readFile_2_Files_match_by_common_primary_key_and_add_a_tokenFrom_SecondFile_to_FirstFile(
																									first_File,
																									delimiter_to_be_added_to_first_File,
																									token_in_first_file_having_primary_Key_CSV,//-1
																									second_File,
																									token_in_second_file_having_primary_Key_CSV,
																									token_interested_in_second_File_to_be_appended_to_first_CSV,
																									token_in_first_file_WHERE_APPENDING_token_interested_in_second_File_to_be_appended_to_first_CSV,
																									outFile_Exists,
																									false, //is_Append_outFile_exists
																									outFile_Not_Exists,
																									false, //is_Append_outFile_not_exists
																									true //add_dummy_tokens_for_NOT_exists
																									);

				System.out.println("first_File:"+first_File);
		      // remove STOP WORDS 
		      // ----out of below method is input for p6_prepare_matrix_author_id_vs_docid_AND_unique_voc_word --->> "inFile_1_all_tokens="
	    	  // NOTE: take output of below file and do grep -v \!\!\!\" to remove the lines with junk delimiter.
	    	  //       and named this file "_REMOVE_STOPWORDS_FINAL.txt"
		      // ALSO REPLACE !!!" to " "
	         readFile_and_remove_stop_words_for_eachLine( 
	        		 									output_both, //allTokens_removed_NOISE, //INPUT
	        		 									"!!!" , // delimiter of  allTokens_removed_NOISE
	        		 									outFile_allTokens_removeSTOPWORDS,
	    		  										index_token_having_bodyText,
		      											false //isSOPprint
		      											);
	         /// 
	         System.out.println("allTokens_removed_NOISE:"+allTokens_removed_NOISE);
	         System.out.println("output_both:"+output_both);
         
	      }
	      
    	  //****IMPORANT NOTE: take output of below file and do grep -v \!\!\!\" to remove the lines with junk delimiter.****
    	  //       and named this file "_REMOVE_STOPWORDS_FINAL.txt"
	      //overwrite
//	      outFile_allTokens_removeSTOPWORDS=  allTokens_removed_NOISE+ "_REMOVE_STOPWORDS_FINAL.txt";
	       
	      
	      //overwrite for "p6" problem to take only 20000 docs
//	      outFile_allTokens_removeSTOPWORDS=baseFolder+"20000.txt";
	      String NOT_matched_file=outFile_allTokens_removeSTOPWORDS+"_NOT_MATCHED.txt";
	      String matched_file=outFile_allTokens_removeSTOPWORDS+"_MATCHED.txt";
//	      outFile_allTokens_removeSTOPWORDS=allTokens_removed_NOISE+"_REMOVE_STOPWORDS_FINAL_bTextNLPtagged.txt_titleNLP.txt";
	      
	      //check the number of tokens in each line 
	      if(Flag==4){
	    	  // THIS WILL ADD DOC ID TO THE END AS LAST TOKEN
	    	  readFile_verify_and_Correct_No_Of_Tokens_In_EachLine.readFile_only_verify_No_Of_Tokens_In_EachLine(
	    			  																	8, //noTokensNeeded, 
	    			  																	"!!!", //delimiter, 
	    			  																	outFile_allTokens_removeSTOPWORDS, //inputFile, 
	    			  																	NOT_matched_file, //outputFile_not_matched, 
	    			  																	matched_file, //outputFile_matched, 
	    			  																	debugFile, 
	    			  																	false, //is_flag_strictly_lesserORequal_noTokensNeeded
	    			  																	true, //is_add_lineNo_at_end
	    			  																	false //isSOPprint
	    			  																	);
	 
	      }
	      //////////////// SKIPPING Flag==44 if ran below ////////////////////////
	      //IMPORTANT: This below is dummy script to produce "20000.txt" - skipping Flag==44 as we wanted already selected URLS "mergedall-2016-T9-10-Topics-all-tokens.txt_ONLY_ENGLISH.txt_removed_NOISE.txt_REMOVE_STOPWORDS_FINAL.txt_MATCHED.txt_rem_DIRTYauthName_remDUPbyURL_BACKUP.txt"
	       
	      
//			readFile_2_Files_match_by_common_primary_key_and_add_a_tokenFrom_SecondFile_to_FirstFile.
//			readFile_2_Files_match_by_common_primary_key_and_add_a_tokenFrom_SecondFile_to_FirstFile(
//																								baseFolder+"mergedall-2016-T9-10-Topics-all-tokens.txt_ONLY_ENGLISH.txt_removed_NOISE.txt_REMOVE_STOPWORDS_FINAL.txt_MATCHED.txt_rem_DIRTYauthName_remDUPbyURL_BACKUP.txt",//second_File,
//																								delimiter_to_be_added_to_first_File,
//																								"2",//token_in_first_file_having_primary_Key_CSV,//-1
//																								baseFolder+"mergedall-2016-T9-10-Topics-all-tokens.txt_ONLY_ENGLISH.txt_removed_NOISE.txt_REMOVE_STOPWORDS_FINAL.txt_MATCHED.txt", //first_file
//																								"2",//token_in_second_file_having_primary_Key_CSV,
//																								"-1",//token_interested_in_second_File_to_be_appended_to_first_CSV,
//																								"-1",//token_in_first_file_WHERE_APPENDING_token_interested_in_second_File_to_be_appended_to_first_CSV,
//																								baseFolder+"20000_fromBACKUP_Exists.txt", //EXISTS
//																								false, //is_Append_outFile_exists
//																								baseFolder+"20000_fromBACKUP_NOTExists.txt", //outFile_Not_Exists,
//																								false, //is_Append_outFile_not_exists
//																								true //add_dummy_tokens_for_NOT_exists
//																								);
	      
	      
	       String all_unique_matched_lines_on_allQUERIES=baseFolder+"20000.txt"; //usually 20000.txt
	       String all_unique_matched_lines_on_allQUERIES_addMISSINGtopic_AND_sourceCHANNEL=baseFolder+"20000_add2MOREfeature.txt";
	       String all_unique_matched_lines_on_allQUERIES_merge_add2MOREfeature=baseFolder+"20000_merge_add2MOREfeature.txt"; //usually 20000.txt
	      //note: use for input to Flag 5 (1) sort_by_datetime_for_extracted_CSV_news_articles
	      //		  					  (2) match_by_set_of_string_onEachLine_write_2_individualFile
	      if(Flag==44 || Flag==999){
	    	   //overwriting for p6 problem <--------------------
//	    	   int index_of_token_having_primarykey=2;  
	    	   int index_of_token_datetime_in_inputFile=3;
	    	   String output_44=matched_file+"_date_SORTED.txt";
		       
		        //sort_by_datetime_for_extracted_CSV_news_articles (comment/uncomment)
	    	  	sort_by_datetime_for_extracted_CSV_news_articles.
		        sort_by_datetime_for_extracted_CSV_news_articles(
		        													baseFolder,
		        													matched_file,
		        													output_44, 	// OUTPUT
		        													index_of_token_datetime_in_inputFile
		        													);
	    	  	 
	            //NOTE: usually we use "sort_by_datetime_for_extracted_CSV_news_articles" and we sort a file by date (a token in a file)..
	            //then the sorted OUTPUT file is given as input to this class match_by_set_of_string_onEachLine_write_2_individualFile. 
	      		//comment this  -->q_topic_10_CSV_string = "crime, india ";
	    	   
	            // match_by_set_of_string_onEachLine_write_2_individualFile 
	            match_by_set_of_string_onEachLine_write_2_individualFile.
	            match_by_set_of_string_onEachLine_write_2_individualFile(
	            														baseFolder,
	            														output_44, //SORTED FILE FORM "sort_by_datetime_for_extracted_CSV_news_articles"
	            														q_topic_10_CSV_string,
	            														index_token_to_Verify_eachLine_token,
	            														token_containing_sourceURL,
	            														index_of_token_having_authName,
	            														top_N_to_be_matched_for_eachQuery,
	            														inFile10_dirty_authNames,
	            														all_unique_matched_lines_on_allQUERIES
	            													);
	    	   

	            
	            //REMOVE Duplicates by sourceURL name..
	            System.out.println("NOTE: go to folder -> "+baseFolder+" 20000.txt auto generated and stored \" ");
//	    	  	System.out.println("-----note: before running 444 , set matched_file manually...");
	      }
	      
	      // (TIME:7minutes for 12k documents)
	      if(Flag==440|| Flag==999){
	    	  
	            // apply missing keyword to each documents
	            applying_missing_keywords_AND_sourcePOLIorTRAD(
	            											baseFolder,
					            							all_unique_matched_lines_on_allQUERIES, //INPUT
					            							all_unique_matched_lines_on_allQUERIES_addMISSINGtopic_AND_sourceCHANNEL,//OUTPUT
					            							token_containing_keywords,
					            							token_containing_sourceFile_TRAD_or_POLI,
					            							q_topic_10_CSV_string
	            						  					);
	            System.out.println("INPUT all_unique_matched_lines_on_allQUERIES:"+all_unique_matched_lines_on_allQUERIES);
	    	  
	      }
	      
	      //if(Flag==4401) -- merge out of applying_missing_keywords_AND_sourcePOLIorTRAD() (file=20000_add2MOREfeature.txt) 
	      // AND file var "all_unique_matched_lines_on_allQUERIES"
	      if(Flag==4401|| Flag==999){
	    	  //merge_missing_keywordFILE_with_allQueriesFILE
	    	  merge_missing_keywordFILE_with_allQueriesFILE(
	    			  											all_unique_matched_lines_on_allQUERIES, //IN 
	    			  											all_unique_matched_lines_on_allQUERIES_addMISSINGtopic_AND_sourceCHANNEL, //IN
	    			  											all_unique_matched_lines_on_allQUERIES_merge_add2MOREfeature //OUT
	    			  										);
	    	  System.out.println("------------------------------------------------");
	    	  System.out.println("NOTE: Before going next, use config mentioned in \"Flag==4402\". YOU have to manuall run the CRAWLER for file in variable"
	    	  					 + " \"input_crawled_file\":"
	    			  			 +all_unique_matched_lines_on_allQUERIES_merge_add2MOREfeature);
	    	  
	      } 
	      
	      
	      // manual change (BELOW IS THE CRAWLED FILE using run crawler.java - "main" option "1" ..(use below config.txt))
	      String input_crawled_file=input_crawled_FOLDER+"3rd_time_url_to_crawl_removedirtyRSSfeed_crawledBODYTEXT.txt";
	      String crawled_matched_file=input_crawled_file+"_matched.txt";
	      		 crawled_matched_file="/Users/lenin/Downloads/Google Drive/##***Now-Working/#Problems/Problem-25-NewsFeed-and-Twitter/Data.p25/outputFolderForUNIQUE_url/crawled_GLOBAL.txt";
	      String crawled_notmatched_file=input_crawled_file+"_notmatched.txt";
	      String OUTPUT_crawled_file_CLEANED=crawled_matched_file+"_CLEANED.txt";
	      String file4_DomainName_NewsStartPatterns="/Users/lenin/Dropbox/#problems/p18/ds1/global_domainNAMES_patternFORNEWSBODYSTART.txt";

	      
	      /// (Flag==4402) -- DO THE CRAWLING HERE..
	      if(Flag==4402){
	    	  ///   run crawler.java - "main" option "1" ..(use below config.txt) 
//	    	  		  NLPfolder=/Users/lenin/OneDrive/jar/NLP/models/
//	    			  inputSingleURL=http://www.yorkregion.com/news-story/4116596-health-hazards-at-daycare-where-girl-died/
//	    			  isAppend=true
//	    			  delimiter=!!!
//	    			  crawlerType=type2Crawler
//	    			  crawlerType#remarks={type1Crawler,type2Crawler,all};type1 uses jericho; type2 uses http
//	    			  inputType=file
//	    			  inputType#={"mongodburlandbody","file"}
//	    			  outputType=writeCrawledText2OutFile
//	    			  outputType#={"writeCrawledText2OutFile","mongodb","outNewUnCrawledURLFile"}
//	    			  flag2=
//	    			  flag2.remarks= You can do set of tagging of NLP such as location, date, person etc.
//	    			  folder=/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/
//	    			  inputListOf_URLfileCSV=20000_merge_add2MOREfeature.txt
//	    			  inputListOf_URLfileCSV#remark=CrawledOutHTML1.txt!!!CrawledOutHTML.txt
//	    			  fromLine=0
//	    			  toLine=500000
//	    			  URL_present_in_which_column_token=2
//	    			  URL_present_in_which_column_token#remark=Starts from 1
//	    			  outputFile=20000_merge_add2MOREfeature_crawledBODYTEXT.txt
//	    			  outdebugErrorAndURLfile=outdebugErrorAndURLfile..1.txt
//	    			  outAlreadyCrawledURLsINaFile=20000_merge_add2MOREfeature_crawledBODYTEXT.txt
//	    			  outdebugErrorAndURL=outdebugErrorAndURL.1.txt
//	    			  outNewUnCrawledURLFile=outNewUnCrawledURLFile.1.txt
//	    			  parseType=noparse
//	    			  isSOPdebug=false
//	    			  is_overwrite_flag_with_flag_model=true
//	    			  is_overwrite_flag_with_flag_model.remark=if true, POSModel passed via param will be used.
//	    			  in_Already_Crawled_File_URL_present_in_which_column_token=1
//	    			  in_Already_Crawled_File_URL_present_in_which_column_token#remark=Starts from 1
//	    			  method=readparsexmlfeeds
	    	  
	  		//verification if all lines have same number of tokens.
//	    	readFile_verify_and_Correct_No_Of_Tokens_In_EachLine.
//	  		readFile_only_verify_No_Of_Tokens_In_EachLine
//	  												 ( 2, //noTokensNeeded
//	  												 "!!!", // delimiter  
//	  												 input_crawled_file,  
//	  												 crawled_matched_file, // outputFile_not_matched,
//	  												 crawled_notmatched_file, //outputFile_matched,
//	  								 	 			 debugFile,
//	  								 	 			 false, //is_flag_strictly_lesserORequal_noTokensNeeded
//	  								 	 			 true, //is_add_lineNo_at_end
//	  								 	 			 false //isSOPprint
//	  								 	 			 );
//	    	  
	    	   // CLEANING CRAWLING DOCUMENTS 
	    	  do_clean_crawling_docs_using_domainNAMES_patternFORNEWSBODYSTART(
	    			  													crawled_matched_file, // IN 
	    			  													file4_DomainName_NewsStartPatterns, //IN
	    			  													OUTPUT_crawled_file_CLEANED //OUT
	    			  	 												);
	    	  
	    	  //clean the URL of the file in second_File (mostly OBSELETE, unless used again as per scenario)
//	    	  clean_url_second_File(second_File,
//	    			  				second_File_cleaned);
	    	  
	    	  
	      }
	      
	      //all_unique_matched_lines_on_allQUERIES_merge_add2MOREfeature, // INPUT
	      
	      // preparing inFile_organiz_person_locations_numbers_docID
	      if(Flag==444|| Flag==999){
	    	  	String inFile_alreadyRan_pastTagging="";
				String NLPfolder="/Users/lenin/OneDrive/jar/NLP/models/";
				boolean isSOPdebug=false;
				String inFile100=all_unique_matched_lines_on_allQUERIES; 
				String outFile=matched_file+"_bTextNLPtagged.txt";
	      		String debugFileName="debug.trad.tagging.txt";
//	      		int token_containing_body_text=8;
//	      		int token_containing_title=6;
//	      		int token_containing_auth_Name=4;
//	      		int token_containing_sourceURL=2;
//	      		int token_containing_keywords=5;
//	      		int token_containing_date=3;
	      		String start_pattern_containing_author_name="";
	      		boolean is_Append_outFile=true;
	      		int skip_top_N_lines=-1;
	      		String Flag_2="dummy";
	      		int Flag_3_feature_parser_types=2;
	      		
	    		// STEP 2: (a) run for bodyText
	    		POSModel inflag_model2 = new POSModelLoader().load(new File(NLPfolder+ "en-pos-maxent.bin"));
	    		////
	    		readFile_pick_a_Token_doNLPtagging_writeOUT.
	    		readFile_pick_a_Token_doNLPtagging_writeOUT(
	    															baseFolder,
	    															token_containing_body_text_WITH_STOPWORDS, //token_position_to_insert_lineNumber,
	    															inFile100, 
	    															outFile, 
	    															inFile_alreadyRan_pastTagging,
	    															1, //default_start_lineNo, 
	    															NLPfolder, 
	    															inflag_model2, 
	    															true, //is_overwrite_flag_with_flag_model, 
	    															isSOPdebug);

	    			// STEP 2: (b) run for TITLE
	    			String outFile_2 = outFile+"_titleNLP.txt";
//	    			/////////
	    			readFile_pick_a_Token_doNLPtagging_writeOUT.
	    			readFile_pick_a_Token_doNLPtagging_writeOUT(
	    																baseFolder,
	    																token_containing_title, //<TITLE>  token_position_to_insert_lineNumber,
	    																outFile,// IN 
	    																outFile_2,  //OUT
	    																inFile_alreadyRan_pastTagging, //IN
	    																1, //default_start_lineNo, 
	    																NLPfolder, 
	    																inflag_model2, 
	    																true, //is_overwrite_flag_with_flag_model, 
	    																isSOPdebug);
	    			
	    			// STEP 3 : 
	        		TreeMap<Integer,TreeMap<Integer, String>> mapOut=new TreeMap<Integer, TreeMap<Integer,String>>();
	        				
//	        	    String NLPfolder="/Users/lenin/OneDrive/jar/NLP/models/";
//	        		String baseFolder="/Users/lenin/Downloads/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/";
//	        		String textNLPtagged_input_2="u.s._DT defense_NN secretary_NN chuck_NN hagel_NN said_VBD on_IN tuesday_IN he_PRP expected_VBD to_TO discuss_VB lessons_NNS learned_VBN from_IN the_DT search_NN for_IN a_DT missing_VBG malaysian_JJ jetliner_NN at_IN talks_NNS with_IN southeast_JJ asian_JJ defense_NN chiefs,_NN but_CC stopped_VBD short_RB of_IN criticizing_JJ malaysia's_NN coordination_NN effort.";
	        		String textNLPtagged_input_2="";
	        	    String intext_NLP_untagged_input_2="";
	        		    		
//	                String inFile_input_1=baseFolder+"mergedall-2016-T9-10-Topics-all-tokens.txt_ONLY_ENGLISH.txt_removed_NOISE.txt_REMOVE_STOPWORDS_FINAL.txt";
//	                	   inFile_input_1="";
	                
	                String 	outFile2=baseFolder+"Noun_count.txt";
	                 
	                int 	token_having_body_text_taggedNLP_inFile=10;
	        				InputStream is = null;  ParserModel inflag_model2_chunking = null;
	        		
	        		String serializedClassifier="/Users/lenin/OneDrive/jar/stanford-ner-2014-01-04/classifiers/english.conll.4class.distsim.crf.ser.gz";
	        		 CRFClassifier<CoreLabel>  NER_classifier=CRFClassifier.getClassifierNoExceptions(serializedClassifier);
	        		
	        		 String debug_File=baseFolder+"debug.txt";
	        		 
	        		try{is=new FileInputStream(NLPfolder+"en-parser-chunking.bin");
	        			inflag_model2_chunking=new ParserModel(is);
	        		}
	        		catch(Exception e){ e.printStackTrace();}
	        		int run_top_N_lines_only=-1;// *******
	                // find_VERB_AND_Noun_from_TaggedString_inFile
	        		 mapOut= find_Verb_AND_Noun_JustBfr_from_TaggedString_inFile.
	        				 find_VERB_AND_Noun_JustBfr_from_TaggedString_inFile(						 baseFolder,
	        				 																			 outFile_2,
	    											                                                     textNLPtagged_input_2, //blank here
	    											                                                     intext_NLP_untagged_input_2,
	    											                                                     outFile2,
	    											                                                     token_having_body_text_taggedNLP_inFile,
	    											                                                     false, //isSOPprint
	    											                                                     inflag_model2_chunking,
	    											                                                     NER_classifier,
	    											                                                     run_top_N_lines_only, //-1 allowed 
	    											                                                     debug_File
	    											                             );
	        		 
	        		 System.out.println("---output---");
	        		 for(int i:mapOut.keySet()){
	        			 if(mapOut.containsKey(i) && mapOut.get(i)!=null){
	    	    			 for(int j:mapOut.get(i).keySet()){
	    	    				 System.out.println(i+"<-->"+ mapOut.get(i).get(j)+"<-->"
	    	    						 +  mapOut.get(i).get(j).replace("!!!", "!#!#").split("!#!#").length  +   "<-->"
	    	    						 + mapOut.get(i).get(j)+"<-->"+mapOut.get(i).size());
	    	    			 }
	        			 }
	        		 }
	        		  
	        		 // OVERWRITE IT ..
	        		 inFile100=outFile_2;
	        		 int token_containing_body_text_added_NLPbodytext_NLPtitle=9; 
	        		 int token_containing_body_text_WITH_NLP=10;
//	        		 int token_containing_primary_key=2;
	        		 
	        		 System.out.println("token_containing_body_text:"+token_containing_body_text_WITH_STOPWORDS+" inFile:"+inFile100 );
	        		 String output_organization_person_count=baseFolder+"output_organiz_person_locations_numbers_docID.txt";
	     
	        		 write_debugFile_MAIN.append("\n ----get_organization_person_location_numbers-----\n"
	        				 					+" inFile100:"+inFile100
	        				 					+" token_containing_body_text_WITH_STOPWORDS:"+token_containing_body_text_WITH_STOPWORDS
	        				 					+" token_containing_body_text_WITH_NLP:"+token_containing_body_text_WITH_NLP
	        				 					+" token_containing_sourceURL:"+token_containing_sourceURL
	        				 					+"\n");
	        		 write_debugFile_MAIN.flush();
	        		 
	        		 //note: INPUT OF THIS METHOD IS OUTPUT OF readFile_pick_a_Token_doNLPtagging_writeOUT() ABOVE
	     		    // STEP: run and get Organizations and Person names mentioned in the token?
	        		 find_Verb_AND_Noun_JustBfr_from_TaggedString_inFile.
	        		 get_organization_person_location_numbers(
	    						 		    					baseFolder,
	    						 		    					inFile100, // has all tokens
	    						 		    					token_containing_body_text_added_NLPbodytext_NLPtitle,
	    						 		    					token_containing_body_text_WITH_NLP,
	    						 		    					token_containing_sourceURL,
	    						 		    					output_organization_person_count,
	    						 		    					-1 // skip_top_N_lines
	     		    										);
	        		 
	        		   
	      }
	      
	      ///////
	      if(Flag==5 || Flag==999){
	    	  //note: use below for cleaning the junk characters.
	    	  // tr -dc [:alnum:][\ ,.]\\n < ./inputfile.txt > ./inputfilefixed.txt
	    	  
	    	  System.out.println("-----matched_file:"+matched_file);
	    	  int token_containing_missingKEYWORDS=10;
	    	  int token_containing_bool_POLIorTRAD=11;
	    	  
	    	  //NOTE10: have to run the below method twice with possible values of run_for_only_inFile10_dirty_authNames
	    	  //(1) with run_for_only_inFile10_dirty_authNames==true and
	    	  //(2) related with (1) then run "cat * > all.txt" from command prompt in folder "dirtyAuthorName"
	    	  //(3) ----OBSOLETE---run around 3 methods in main of find_Verb_AND_Noun_JustBfr_from_TaggedString_inFile N get "inFile_organiz_person_locations_numbers_docID.txt"
	    	  //		with "20000.txt" as input.. configuration file has below two parameters 
	    	  //inFile=20000.txt  (see "find_Verb_AND_Noun_JustBfr_from_TaggedString_inFile" for complete config) 
	    	  //outFile=20000.txt_bTextNLPtagged.txt
	    	  //(4) with run_for_only_inFile10_dirty_authNames==false for conv_2_mallet_format
	    	  conv_2_mallet_format(
	    			  			  baseFolder,
	    			  			  all_unique_matched_lines_on_allQUERIES_merge_add2MOREfeature, //INPUT ( have right number of tokens ) <- The docID IN output MALLET file is actually from this FILE..
				    			  "!!!", //outFile_allTokens_removeSTOPWORDS
		    			  		  inFile_organiz_person_locations_numbers_docID, // run around 3 methods in main of find_Verb_AND_Noun_JustBfr_from_TaggedString_inFile N get "inFile_organiz_person_locations_numbers_docID.txt" 
		    			  		  "@@@@", //delimiter_for_inFile_organiz_person_locations_numbers_docID
				    			  1, //index_of_token_having_bodyText(outFile_allTokens_removeSTOPWORDS),
				    			  token_containing_title, //6, //index_of_token_having_title
				    			  token_containing_keywords, //5, //index_of_token_having_keywords
								  9, //index_of_token_having_docID,
								  token_containing_sourceURL, //2, //index_of_token_having_primarykey,
								  index_of_token_having_authName, //index_of_token_having_authName
								  token_containing_sourceFile_TRAD_or_POLI, //index_of_token_having_sourceFileName
								  token_containing_missingKEYWORDS,
								  token_containing_bool_POLIorTRAD,
								  inFile10_dirty_authNames, // INPUT
								  matched_file+"_4_mallet_DEBUG.txt",  //OUTPUT
				    			  matched_file+"_rem_DIRTYauthName_remDUPbyURL_DEBUG.txt", // outFile_removed_DIRTY_authName -- OUTPUT
				    			  matched_file+"_4_PLSI_input_only_bodyText_DEBUG.txt", //outFile_for_PLSI_input_only_bodyText
				    			  is_run_for_only_inFile10_dirty_authNames,
				    			  replace_string_for_sourceChannelFilePath
	    			  			 );	    	  			
				    	  ///////////////
	    	  			  System.out.println("----------------------------------------------------------------");
				    	  System.out.println("input file :"+matched_file);
				    	  System.out.println("MALLET output file :"+matched_file+"_4_mallet.txt");
				    	  System.out.println("output file 2:"+ matched_file+"_rem_DIRTYauthName_remDUPbyURL.txt");
				    	  System.out.println("Debug file:"+baseFolder+"debug_cleaning_and_convert.txt");
				    	  System.out.println("NOTE: check if "+matched_file+"_rem_DIRTYauthName_remDUPbyURL.txt" +" and "+matched_file+"_4_mallet.txt"+" has same number of lines");
				    	  
				    	  if(is_run_for_only_inFile10_dirty_authNames==true){
				    		  System.out.println("----------------------------------------------------------------");
				    		  System.out.println("NOTE: run--> cat *> all.txt in folder "+baseFolder+"dirtyAuthorName as U just ran with is_run_for_only_inFile10_dirty_authNames==true");
				    	  }
				    	  
	      }
	      
         //---------------> next run p6_MAIN_Work()  --------------->
 
//        inFile="/Users/lenin/Downloads/p6/merged/dummy.mergeall/authornameextract2/mergedall-2016-10topics-english-only-bodyText-token.txt";
//        outFile_only_int_token="/Users/lenin/Downloads/p6/merged/dummy.mergeall/authornameextract2/mergedall-2016-10topics-english-only-Numbers.txt";
//        EXTRACT NUMBERS
//        readFile_and_get_only_numbers_from_eachLine(
//								        		inFile,
//								        		outFile_only_int_token
//				        		);


//         
//      String inputFile="/Users/lenin/Downloads/p6/merged/dummy.mergeall/Merged.Main/mergedall-2016-all-tokens.txt";
//	    String outFile_1="/Users/lenin/Downloads/p6/merged/dummy.mergeall/ds8/mergedall-2016-T9-10-Topics-all-tokens.txt"; //get all tokens
//	    String outFile_2="/Users/lenin/Downloads/p6/merged/dummy.mergeall/ds8/mergedall-2016-T9-10-Topics-only-bodyText.txt"; //get only bodyText
//		int    interested_token_for_bodyText_inputFile=8;
//		//***OBSOLETE as merged to method "find_get_a_token_match_on_N_Topics_and_write_to_another"
//		find_and_filter_Given_10_topics(inputFile, outFile_1, outFile_2, 
//										interested_token_for_bodyText_inputFile);
//		
 
		//System.out.println("english only documents:"+mapOutStat.get(2)+" out of total docs:"+mapOutStat.get(1));
		
	    System.out.println("NOTE: use grep -v |||\" to remove lines that that junk delimiter");
		System.out.println("Time Taken:" + NANOSECONDS.toSeconds(System.nanoTime() - t0) +" seconds; "
       	   				  + (NANOSECONDS.toSeconds(System.nanoTime() - t0))/60 +" minutes");
         
	      // match and get cleaned bodyText
	      String inFile_1=baseFolder+"mergedall-2016-T9-10-Topics-all-tokens.txt";
	      String inFile_2=baseFolder+"mergedall-2016-all-token-3-Sep-to-26-Oct-T9-.space-10-Topics.txt";
	      String output_matchedFile=baseFolder+"mergedall-2016-all-token_matched.txt";
	      //*****OBSELETE (AWK handling) match and get new bodyText (having space between 2 tokens)
//	      match_get_bodyText_for_spaceAdded(  baseFolder,
//	    		  							  inFile_1,
//	    		  						      inFile_2,
//	    		  						      output_matchedFile//this file format similar as inFile_1
//	    		  							);
	      
	    }
	    catch(Exception e){
	    	e.printStackTrace();
	    }
		
	}

	


}
