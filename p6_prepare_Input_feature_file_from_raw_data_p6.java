

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.io.StringWriter;
import java.sql.Clob;
import java.util.TreeMap;

import crawler.convert_givenString_to_Json;
import crawler.crawler;
import opennlp.tools.parser.ParserModel;
import opennlp.tools.postag.POSModel;

// PREPARE INPUT prepare_Input_feature_file_from_raw_data FOR p6
// Each line in given input RAW data is a document
public class p6_prepare_Input_feature_file_from_raw_data_p6 {

	//NOTE: not need NLP , THEN use "readFile_readEachLine_For_a_Set_of_Start_String_End_String_WriteToAnotherFile" to extract
	// NLP** tagging for title and bodyText 
	// extract (1) authors, title, body_text (2) tagged numbers, people names and extract them 
	// each line contains <document text, title, author names, etc> for a document  <> inFile_containing_Authors_AND_Documents
	// line no of a document = UNIQUE Doc ID added in output File
	// outFile_intermediate => <Doc_ID,AuthorNames,Title,Body_text,>
	// NOTE: use "readFile_pick_a_Token_doNLPtagging_writeOUT" for only to do NLP TAGGING
	public static void extract_features_from_Each_doc(
						  				String 		inFile_containing_Authors_AND_Documents,
						  				String 		outFile_intermediate,
						  				boolean 	is_Append_outFile_intermediate,
						  				String	    start_pattern_containing_author_name, //example: from:
						  				int			token_containing_body_text, //starts with 1
						  				int         token_containing_title,    //starts with 1
						  				int			token_containing_auth_Name,
						  				int   		token_containing_sourceURL,
						  				int         token_containing_keywords,
						  				int         token_containing_date,
						  				String      NLPfolder, //NLP model folder
						  				String  	debugFileName,
						  				boolean     is_SOP_print,
						  				int 		skip_top_N_lines,
						  				String      Flag_2,
						  				int	      	Flag_3_feature_parser_types
						  				){
		BufferedReader br=null;
		TreeMap<String, String> maptaggedNLP=new TreeMap<String, String>(); 
		TreeMap<Integer, Integer> map_as_given_index_of_token_as_KEY=new TreeMap<Integer, Integer>();
		map_as_given_index_of_token_as_KEY.put(token_containing_body_text, -1);
		map_as_given_index_of_token_as_KEY.put(token_containing_title, -1);
		map_as_given_index_of_token_as_KEY.put(token_containing_auth_Name, -1);
		map_as_given_index_of_token_as_KEY.put(token_containing_sourceURL, -1);
		map_as_given_index_of_token_as_KEY.put(token_containing_keywords, -1);
		map_as_given_index_of_token_as_KEY.put(token_containing_date, -1);
		
		try{
			br = new BufferedReader(new FileReader(inFile_containing_Authors_AND_Documents));
			FileWriter writer= new FileWriter(new File(outFile_intermediate), is_Append_outFile_intermediate);
			
			FileWriter writer_debug= new FileWriter(new File(debugFileName), true);
			
			String line=""; int lineNumber=0;int orig_lineNumber=0;
			String author_names=""; String body_text=""; String title="";
			int begin_index = -1; int end_index=-1;
			String curr_title_NLPtagged="";
			String curr_body_text_NLPtagged="";
			int debug_count_blank_lines=0;
			String keywords="", messageId="", date="";
			String sourceURL="";
			//HEADER
			writer.append("lineNo!!!authName!!!keywords!!!date!!!sourceURL!!!messageID!!!Title!!!Title_NLPtagged!!!"
							+ "bodyText!!!bodyText_NLPtagged!!!otherTokens...\n");
			writer.flush();
			
			//each line
			while( (line = br.readLine()) !=null){
				keywords=""; messageId=""; date="";
				author_names=""; body_text=""; title="";
				sourceURL="";
				
				//clean inconsistent delimiter
				line=line.replace("!!!!!!","!!!d!!!").replace("!!!!!", "!!!").replace("!!!!", "!!!");
				
				line=line.toLowerCase();
				int end2=line.length();
				orig_lineNumber++;
				if(skip_top_N_lines>0 && orig_lineNumber<=skip_top_N_lines){
					System.out.println("skipping line->"+orig_lineNumber +" (skip_top_N_lines:"
									    +skip_top_N_lines+")");
					continue;
				}
				//System.out.println(" skipping line->"+orig_lineNumber);
				if(is_SOP_print)
					System.out.println("l:"+line);
				
				//
				if(line.length()>=1){
					String [] tokens=line.split("!!!");
					//no token
					if(tokens.length==0 ){
						debug_count_blank_lines++;
						continue;
					}
					if(start_pattern_containing_author_name==null ){
						start_pattern_containing_author_name="";
					}
					
					lineNumber++;
					
					//parser type
					if(Flag_3_feature_parser_types==1){
					
					if(start_pattern_containing_author_name.length()>0){
						begin_index=line.indexOf(start_pattern_containing_author_name);
						end_index=line.indexOf("!!!", 
												begin_index+start_pattern_containing_author_name.length()
												);
						//
						if(begin_index>=0 && end_index>0 && end_index>begin_index)
							author_names=line.substring(begin_index,end_index).replace("mime-version", "");
						else{
							author_names="";
						}
					}
					else{
						author_names=tokens[token_containing_auth_Name-1];
					}
					
					
					try{
						body_text=tokens[token_containing_body_text-1];
						//RESET IF NOT BODY
						if(body_text.indexOf("fz:")>=0 || body_text.indexOf("urn:")>=0 || body_text.indexOf("date:")>= 0
							|| body_text.indexOf("from:")>=0 || body_text.indexOf("content-base:")>=0
							|| body_text.indexOf("from -")>=0 || body_text.indexOf("message-id:")>=0
							|| body_text.indexOf("infile:")>=0
							){
							body_text="";
						}
					}
					catch(Exception e){
						body_text="";
					}
					try{
						title=tokens[token_containing_title-1];
						//reset if NOT TITLE
						if(title.indexOf("date:")>= 0 || title.indexOf("from:")>=0
							|| title.indexOf("from:")>=0 ||title.indexOf("content-base:")>=0
							|| title.indexOf("from -")>=0 || title.indexOf("message-id:")>=0
							|| title.indexOf("infile:")>=0
								){
							title="";
						}
						
					}
					catch(Exception e){
						title="";
					}
					 
					//keywords
					try{
						int begin=line.indexOf("keywords:");
						int end=line.indexOf("!!!",begin+8);
	
						if(begin>=0 && end==-1)
							end=end2;
						
						if(begin>=0 && end >=0 && end>begin){
							keywords=line.substring(begin, end);
						}
						else{
							keywords="err:"+begin+"::"+end2+"::"+end;
						}
							
					}
					catch(Exception e){
					
					}
					 
					//
					try{
						int begin=line.indexOf("date:");
						int end=line.indexOf("!!!",begin+4);
						if(begin>=0 && end==-1)
							end=end2;
						
						if( begin >=0 && end>=0 && end>begin){
							date=line.substring(begin, end);
						}
						else{
							date="";
						}
							
					}
					catch(Exception e){
					
					}
					 
					// message id 
					try{
						int begin=line.indexOf("message-id:");
						int end=line.indexOf("!!!", begin+9);
						 
						if(begin>=0 && end==-1)
							end=end2;
						
						if( begin >=0 && end>=0  && end>begin){
							messageId=line.substring(begin, end);
						}
						else{
							messageId="";
						}
							
					}
					catch(Exception e){
					
					}
					
					
					} //END if(Flag_3_feature_parser_types=1){
					
					
					// 
					if(Flag_3_feature_parser_types==2){
						
						try{
						
//						messageId=tokens[token_containing__message_id];
						keywords=tokens[token_containing_keywords-1];
						date=tokens[token_containing_date-1];
						author_names=tokens[token_containing_auth_Name-1];
						body_text=tokens[token_containing_body_text-1];
						sourceURL=tokens[token_containing_sourceURL-1];
						title=tokens[token_containing_title-1];
						
						}
						catch(Exception  e){
							writer_debug.append("\n error on count : lineNumber:"+lineNumber +" sz:"+tokens.length);
							writer_debug.flush();
						}
					}
					
					// other TOKEN  not given as index in input param
					int cnt=0; String conc_other_tokens="";
					while(cnt<tokens.length){

						int token_index=cnt+1;
						if( map_as_given_index_of_token_as_KEY.containsKey(token_index) ){
						
							if(conc_other_tokens.length()==0){
								conc_other_tokens=tokens[cnt];
							}
							else{
								conc_other_tokens=conc_other_tokens+"!!!"+tokens[cnt];	
							}
							
						}
						cnt++;
					}
					
					
//					try{
//						if(title.length()>1){ //NOT NULL
//							
//							System.out.println(" %%%%%%%%%%%%%%%%%%% START calling for title lineNumber:"+lineNumber);
//							maptaggedNLP=crawler.wrapper_getNLPTrained_en_pos_maxent(
//					 							"", //give input_1 or input_2
//					 							title,
//										  		NLPfolder,
//											    "en-pos-maxent.bin", // Flag={"en-pos-maxent.bin","en-ner-person.bin"}
//											    Flag_2,
//										  		is_SOP_print, //  isSOPdebug,
//										  		debugFileName
//										  		);
//							if(is_SOP_print)
//								System.out.println("##### maptaggedNLP.size:"+maptaggedNLP.size()+" =>title");
//							//curr_title_NLPtagged=maptaggedNLP.get("taggedNLP");
//							curr_title_NLPtagged="!!!title.feature.tags:"+maptaggedNLP.get("taggedNLP");
//							
//							if(is_SOP_print)
//								System.out.println(" %%%%%%%%%%%%%%%%%%% END calling for title");
//						}
//						else{
//							curr_title_NLPtagged="";
//						}
//					}
//					catch(Exception e){
//						e.printStackTrace();
//					}
					//BOTH TITLE + BODYTEXT
					if(body_text.length()>1){ // NOT NULL
						if(is_SOP_print)
							System.out.println(" *************** START calling for TITLE + body_text");
						maptaggedNLP=crawler.wrapper_getNLPTrained_en_pos_maxent(
													 							"", //give input_1 or input_2
													 							title +".."+body_text,
																		  		NLPfolder,
																		  		"en-pos-maxent.bin", // Flag={"en-pos-maxent.bin","en-ner-person.bin"}
																		  		Flag_2, //Flag_2
																		  		is_SOP_print, //  isSOPdebug,
																		  		false, //is_overwrite_flag_with_flag_model
																		  		null, //inflag_model2
																		  		null,  //inflag_model2_chunking
																		  		debugFileName
																		  		);
						
					
						if(is_SOP_print)
							System.out.println("@@@@@ maptaggedNLP.size:"+maptaggedNLP.size()+" =>bodytext");
						//curr_body_text_NLPtagged=maptaggedNLP.get("taggedNLP");
						curr_body_text_NLPtagged="!!!bodytext.feature.tags:"+maptaggedNLP.get("taggedNLP");
						 
						if(is_SOP_print)
							System.out.println(" *************** END calling for body_text");
					}
					else{
						curr_body_text_NLPtagged="";
					}
					
					
					
					if(is_SOP_print)
						System.out.println("authors="+author_names+" title="+title+" doc_text="+body_text);
					
					//
					writer.append(orig_lineNumber 
									+"!!!"+author_names
									+"!!!"+keywords
									+"!!!"+date
									+"!!!"+sourceURL
									+"!!!"+messageId
									+"!!!"+ title
									+"!!!"+curr_title_NLPtagged 
									+"!!!"+body_text
									+"!!!"+curr_body_text_NLPtagged
									+"!!!"+conc_other_tokens
									+"\n");
					writer.flush();
					
				}
				
				System.out.println("debug_count_blank_lines:"+debug_count_blank_lines);
				//writer.append( line.substring(1, line.length()) +"\n");
				
			} 
			
		}
		catch(Exception e){
			e.printStackTrace();
		}
		
	}
	 
	// create matrix of auth_id and doc_id
	// each line contains <document text, title, author names, etc> for a document  <> inFile_containing_Authors_AND_Documents
	// line no of a document = UNIQUE Doc ID
	public static void create_matrix_auth_id_AND_doc_id(
						  				String 		inFile_containing_Authors_AND_Documents,
						  				String 		outFile_intermediate,
						  				boolean 	is_Append_outFile_intermediate,
						  				String	    start_pattern_containing_author_name, //example: from:
						  				int			token_containing_body_text, //starts with 1
						  				int         token_containing_title    //starts with 1
						  				){
		
	}
	
	// create matrix of doc_id and word_id:frequency
	public static void create_matrix_doc_id_AND_word_id(
										String 					inputFile, 
							  			TreeMap<String,String> 	unique_Dictionary
						  						){
		BufferedReader br=null;
		try{
			br = new BufferedReader(new FileReader(inputFile));
			FileWriter writer= new FileWriter(new File(inputFile));
			String line="";
			//each line
			while( (line = br.readLine()) !=null){
				System.out.println("l:"+line);
				//
				if(line.length()>=1){
					
				}
				//writer.append( line.substring(1, line.length()) +"\n");
				writer.append( "mv "+line+" " +line.replace(".g.txt", ".g") +"\n");
				writer.flush();
			}
			System.out.println("tes");
			
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}
	//main
	public static void main(String[] args) throws IOException {
		
		
		/////// STEP 1 :
		
		TreeMap<String,String> mapout= new TreeMap<String, String>();
		String inFile="/Users/lenin/Downloads/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/merged-trad-29-Jun-policy-26Oct.txt";
		String outFile="/Users/lenin/Downloads/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/merged-trad-29-Jun-policy-26Oct_out.txt";
		
		boolean is_Append_outFile=false;
		
		TreeMap<String,String>  mapConfig=new TreeMap<String, String>();
		String	 confFile="/Users/lenin/Dropbox/workspace-sts-3.4.0.RELEASE.macmini/crawler/conf/24.extract_features_from_Each_doc/trad.config.txt";
		//p18 problem
		confFile="/Users/lenin/Dropbox/workspace-sts-3.4.0.RELEASE.macmini/crawler/conf/p.2.extract_features_from_Each_doc/trad.config.txt";
		
//		File file_confFile=new File(confFile);
//		FileWriter writer=new FileWriter(file_confFile);
//		//create this file 
//		if(!file_confFile.exists()){
//			file_confFile.createNewFile();
//		}
		
		mapConfig=crawler.getConfig(confFile);
		 
		TreeMap<String, String> maptaggedNLP=new TreeMap<String, String>(); 
		String  input_ExtractedText_OR_input2="";
		        inFile=mapConfig.get("inFile");
		String  baseFolder=mapConfig.get("baseFolder");
        		inFile=baseFolder+inFile;
        		
        		outFile=mapConfig.get("outFile");
        		outFile=baseFolder+outFile;
		String  NLPfolder=mapConfig.get("NLPfolder");
	    boolean isSOPdebug=Boolean.valueOf(mapConfig.get("isSOPdebug"));
	    String  debugFileName=baseFolder+mapConfig.get("debugFileName"); 
	    int token_containing_body_text= Integer.valueOf(mapConfig.get("token_containing_body_text"));
	    int token_containing_title=Integer.valueOf(mapConfig.get("token_containing_title"));
	    int token_containing_auth_Name=Integer.valueOf(mapConfig.get("token_containing_auth_Name"));
	    int token_containing_sourceURL= Integer.valueOf(mapConfig.get("token_containing_sourceURL"));
		int token_containing_keywords= Integer.valueOf(mapConfig.get("token_containing_keywords"));
		int token_containing_date= Integer.valueOf(mapConfig.get("token_containing_date"));
	    
	    String	start_pattern_containing_author_name=mapConfig.get("start_pattern_containing_author_name"); //from:
	      		is_Append_outFile=Boolean.valueOf(mapConfig.get("is_Append_outFile"));
	    int		skip_top_N_lines= Integer.valueOf(mapConfig.get("skip_top_N_lines"));
	    String  Flag_2=mapConfig.get("Flag_2");
	    int  Flag_3_feature_parser_types=Integer.valueOf(mapConfig.get("Flag_3_feature_parser_types"));
	    
		System.out.println("Given Configuration:");
			//
		for(String i:mapConfig.keySet()){
			System.out.println(i+" "+mapConfig.get(i));
		} 
		
		/////// STEP 1 :
		
		//NOTE: not need NLP , THEN use "readFile_readEachLine_For_a_Set_of_Start_String_End_String_WriteToAnotherFile" to extract
		// extract_features_from_Each_doc ( NLP tagging for title and bodyText )
		// NOTE: find_Verb_AND_Noun_JustBfr_from_TaggedString_inFile() has a full set up for extract_features_from_Each_doc()
		extract_features_from_Each_doc(
										inFile,
										outFile,
										is_Append_outFile, //is_Append_outFile
										start_pattern_containing_author_name, //"from:",
										token_containing_body_text, //body_text
										token_containing_title,  //title
										token_containing_auth_Name,
										token_containing_sourceURL,
										token_containing_keywords,
										token_containing_date,
										NLPfolder, //NLP model folder
										debugFileName,
										isSOPdebug, // false, //	is_SOP_print
										skip_top_N_lines,
										Flag_2,
										Flag_3_feature_parser_types
										 );
 
		System.out.println("clean:" );
		
	}
}
