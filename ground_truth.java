

import static java.util.concurrent.TimeUnit.NANOSECONDS;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.Map.Entry;

import org.apache.commons.io.FileUtils;
import org.apache.poi.hpsf.WritingNotSupportedException;

import com.google.common.math.LongMath;

import crawler.LoadMultipleValueToMap2_AS_KEY_VALUE;
import crawler.find_Verb_AND_Noun_JustBfr_from_TaggedString_inFile;
import crawler.readFile_eachLine_get_a_particular_token_calc_frequency;
import crawler.readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile;
import crawler.sort_given_treemap;
 
//import p8.wrapper_2_graphOUTfile_4_tag_for_person_organiz_locati;

public class ground_truth {

	//
	public static int convert_bucket( int i, FileWriter writerDebug, boolean isDebug ){
		int value=1;
		int multipler=2;
		try{
			if(i>=1 && i<=2)
				value=46*multipler;
			else if(i>=3 && i<=4)
				value=44*multipler;
			else if(i>=5 && i<=6)
				value=42*multipler;
			else if(i>=7 && i<=8)
				value=40*multipler;
			else if(i>=9 && i<=10)
				value=38*multipler;
			else if(i>=11 && i<=12)
				value=36*multipler;
			else if(i>=13 && i<=14)
				value=34*multipler;
			else if(i>=15 && i<=16)
				value=32*multipler;
			else if(i>=17 && i<=18)
				value=30*multipler;
			else if(i>=19 && i<=20)
				value=28*multipler;
			else if(i>=21 && i<=23)
				value=26*multipler;
			else if(i>=24 && i<=25)
				value=24*multipler;
			else if(i>=26 && i<=27)
				value=22*multipler;
			else if(i>=28 && i<=29)
				value=20*multipler;
			else if(i>=30 && i<=31)
				value=18*multipler;
			else if(i>=32 && i<=33)
				value=16*multipler;
			else if(i>=34 && i<=36)
				value=14*multipler;
			else if(i>=37 && i<=39)
				value=12*multipler;
			else if(i>=40 && i<=42)
				value=10*multipler;
			else if(i>=43 && i<=45)
				value=8*multipler;
			else if(i>=46 && i<=48)
				value=6*multipler;
			else if(i>=49 && i<=50)
				value=4*multipler;
			
			if(isDebug){
				writerDebug.append("\n value:"+value+" i="+i);
				writerDebug.flush();
			}
		}
		catch(Exception e){
			e.printStackTrace();
		}
			
		return value;
	}
	// calc ground truth 
	 //TIME: 8 hours for 7800 authors for approach 1,  6 minutes for approach 2(Flag_2_choose_approach_topicDoc_count_for_each_author=2)
	public static boolean calc_ground_truth(
											String baseFolder,
										    String inFile_1_tf_idf_each_doc,
										    String inFile_2_Distinct_Noun_eachLine_as_doc,
										    String inFile_3_with_all_tokens,
										    int    token_no_contain_authors_in_inFile_3,
										    int	   token_no_contain_bodyText_in_inFile_3,
										    String inFile_4_contains_only_bodyText,
										    String inFile_5_AuthorID_Doc_ID_matrix,
										    String inFile_6_AuthorName_AND_Auth_ID,
										    String inFile_7_auth_id_doc_id_queryTopicRelated,
										    String inFile_8_get_organization_person_location_docID,
										    String inFile_10_authName_AND_auth_id__MAPPEDmultiAuthID_SORTED_add_minAuthID,
										    String inputFolderName_inFile_3,
										    String outputFileName_inFile_3_keyword_AND_freq,
										    String outputFileName_onlykeywords_inFile_3,
										    String CSV_containing_10_interested_topics,
										    String output_Ground_Truth,
										    String output_Ground_Truth_removeDuplicate,
										    int    Flag_2_choose_approach_topicDoc_count_for_each_author,
										    boolean is_skip_calling_calc_Frequency_Of_Token_AND_output_unique_values_in_token,
										    String  replace_string_for_sourceChannelFilePath,
										    String debugFile
										){
		
//		int Flag_2_choose_approach_topicDoc_count_for_each_author=1;
//		int Flag_choose_approach_=1;

		int MAX_docID_of_=0;
		TreeMap<Integer,String>map_eachLine_inFile10__MAPPEDmultiAuthID_SORTED_add_minAuthID=new TreeMap<Integer, String>();
		TreeMap<Integer, String> map_eachLine_as_doc_all_tokens=new TreeMap<Integer, String>();
		TreeMap<String, Integer> mapUnique_Authors_Freq=new TreeMap<String, Integer>();
		TreeMap<Integer, Double> map_eachDoc_Unique_Nouns_count=new TreeMap<Integer, Double>();
		String[] given_N_topics=CSV_containing_10_interested_topics.split(",");
		TreeMap<Integer,String> mapSI_Unique_AuthID_AuthName= new TreeMap<Integer, String>();
		boolean is_add_last_token_of_inputFolderName_to_OUTPUT=false;
		TreeMap<String, String> mapUnique_Authors_Freq_EachLine=new TreeMap<String, String>();
		
		FileWriter writerDebug=null;
		TreeMap<Integer, String> map_eachLine_as_doc_bodyText=new TreeMap<Integer, String>();
		TreeMap<String,Integer> mapSI_Unique_AuthName_AuthID= new TreeMap<String, Integer>();
		TreeMap<Integer, String> map_eachLine_as_doc_AuthorName_AND_bodyText=
				new TreeMap<Integer, String>();
		
		 
		TreeMap<String, TreeMap<String, Integer>> map_authorName_N_topic_freq=new TreeMap<String,TreeMap<String, Integer>>();
		TreeMap<Integer, TreeMap<String, Integer>> map_authID_N_QueryTopic_freq=new TreeMap<Integer,TreeMap<String, Integer>>();
		
		TreeMap<Integer, TreeMap<String, Integer>> map_authID_N_QueryTopic_freq_noDupApplyingMINauthID
										=new TreeMap<Integer,TreeMap<String, Integer>>();
		
		//TreeMap<Integer, Integer> map_eachDoc_TF_IDF_Feat_Vect=new TreeMap<Integer, Integer>();
		try{
			writerDebug=new FileWriter(new File(debugFile));
			try{

			// NOTE : SKIP THIS PROCESS IF ALREADY THIS IS RAN OUT OF "calc_ground_truth"
			if(is_skip_calling_calc_Frequency_Of_Token_AND_output_unique_values_in_token==false){
				// STEP 1: GET author names and frequency of occurrence				
				readFile_eachLine_get_a_particular_token_calc_frequency.
				calc_Frequency_Of_Token_AND_output_unique_values_in_token(
																inFile_3_with_all_tokens,
																"!!!", //inFile_3_with_all_tokens
																outputFileName_inFile_3_keyword_AND_freq,
																false, //is_Append_outputFileName,
																outputFileName_onlykeywords_inFile_3,
																false, //is_Append_outputFileName_onlykeywords,
																inFile_1_tf_idf_each_doc,
																4, // index_of_token_interested_for_freq_count
																2, // index_of_token_of_SourceURL, //used only is_token_interested__authors==true
																7, // index_of_token_of_Source_FileName, //used only is_token_interested__authors==true
																replace_string_for_sourceChannelFilePath,
																true, //is_token_interested__authors
																false, //is_split_on_blank_space_in_token
																false, //is_do_stemming_on_word
																"", //"from:", //YES_filter
																"", //"http", // NO_FILTER
																true, //  isMac
																true, //  is_split_CSV_on_the_given_token
																debugFile,
																true, //is_add_last_token_of_inputFolderName_to_OUTPUT
																false         // isSOPprint
																);
			}

			
			}
			catch(Exception e){
				System.out.println("step 1...");
				e.printStackTrace();
			}
			System.out.println("mapUnique_Authors_Freq(size):"+mapUnique_Authors_Freq.size());
			
			try{
				mapUnique_Authors_Freq_EachLine=
				readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
				readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile(
																	outputFileName_inFile_3_keyword_AND_freq,
																	-1, 
																	-1, 
																	"",//outFile  
																	false, //is_Append_outFile, 
																	false,//is_Write_To_OutputFile, 
																	"debug_label", //, 
																	-1, // token for  Primary key
																	false // isSOPprint
																 );
				System.out.println("LOADED mapUnique_Authors_Freq_EachLine:"+mapUnique_Authors_Freq_EachLine.size()
								+" outputFileName_inFile_3_keyword_AND_freq:"+outputFileName_inFile_3_keyword_AND_freq);
				
				
				 /// AuthName!!!AuthID!!!cross-MultiAuthIDmatched!!!distinctCount-cross-MultiAuthIDmatched!!!ALL_authIDs!!!min_AUTHID
				map_eachLine_inFile10__MAPPEDmultiAuthID_SORTED_add_minAuthID=
				readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
				readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
																inFile_10_authName_AND_auth_id__MAPPEDmultiAuthID_SORTED_add_minAuthID,
																-1, 
																-1,
																 "load bodytext",//debug_label,
																 false // isPrintSOP
																);
				
				
			}
			catch(Exception e){
				System.out.println("step 1...");
				e.printStackTrace();
			}
			
			// mapping min_AUTHID and ALL_authIDs
			TreeMap<Integer,String> mapAuthId_authName_inFile10=new TreeMap<Integer, String>();
			TreeMap<Integer,Integer> map_authID_minAuthId_inFile10=new TreeMap<Integer, Integer>();
			//AuthName!!!AuthID!!!cross-MultiAuthIDmatched!!!distinctCount-cross-MultiAuthIDmatched!!!ALL_authIDs!!!min_AUTHID
			for(int seq_2:map_eachLine_inFile10__MAPPEDmultiAuthID_SORTED_add_minAuthID.keySet()){
				if(seq_2==1) continue; //HEADER
				String line10=map_eachLine_inFile10__MAPPEDmultiAuthID_SORTED_add_minAuthID.get(seq_2);
				String [] s =line10.split("!!!");
				String authName=s[0];
				int authID=Integer.valueOf(s[1]);
				int min_authID=Integer.valueOf(s[5]);
				String ALL_authIDs=s[4];
				
				mapAuthId_authName_inFile10.put(authID, authName);
				
				if(ALL_authIDs.indexOf(",")==-1)
					map_authID_minAuthId_inFile10.put( Integer.valueOf(ALL_authIDs)  , min_authID);
				else{
					String [] s2=ALL_authIDs.split(",");
					int c=0;
					while(c < s2.length){
						map_authID_minAuthId_inFile10.put( Integer.valueOf(s2[c])  , min_authID  );	
						c++;
					}
					
				}
				
			}
			
			
			// <authName, freq> from file: "authName_AND_freq.txt"
			for(String curr_auth_freq:mapUnique_Authors_Freq_EachLine.keySet() ){
				String []s=curr_auth_freq.split("!!!");
				String curr_name=p6_MAIN_Work.p6_normalize_auth_name(s[0]).replace("auth:from: ", "").replace("from: ", "")
																			.replace("from:", "");
				try{
					mapUnique_Authors_Freq.put(curr_name,  
												Integer.valueOf(s[1]) );
				}
				catch(Exception e){
					
				}
			}
			System.out.println("mapUnique_Authors_Freq(size):"+mapUnique_Authors_Freq.size());
			TreeMap<String, String> mapUnique_AuthName_AuthID_EachLine=new TreeMap<String, String>();
			
			try{
				// STEP 1.1
				mapUnique_AuthName_AuthID_EachLine=
						readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
						readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile(
													inFile_6_AuthorName_AND_Auth_ID,
													-1, 
													-1, 
													"",//outFile, 
													false, //is_Append_outFile, 
													false,//is_Write_To_OutputFile, 
													"debug_label", //, 
													-1, // token for  Primary key
													false // isSOPprint
													);
				System.out.println("LOADED mapUnique_AuthName_AuthID_EachLine:"+mapUnique_AuthName_AuthID_EachLine+
									" inFile_6_AuthorName_AND_Auth_ID:"+inFile_6_AuthorName_AND_Auth_ID);
			}
			catch(Exception e){
				System.out.println("step 1...");
				e.printStackTrace();
			}
			if(mapUnique_AuthName_AuthID_EachLine.size()==0){
				System.out.println("--- ERROR: load failed mapUnique_AuthName_AuthID_EachLine...");
				return false;
			}
					String curr_sourceChannel="";
			
					// <authName,AuthID> TO <mapSI_Unique_AuthID_AuthName>
					for(String curr_authName_authID:mapUnique_AuthName_AuthID_EachLine.keySet() ){
						String []s=curr_authName_authID.split("!!!");
						String curr_name=p6_MAIN_Work.p6_normalize_auth_name(s[0]).replace("auth:from: ", "");
						mapSI_Unique_AuthName_AuthID.put( curr_name,
														  Integer.valueOf(s[1]) );
						mapSI_Unique_AuthID_AuthName.put( Integer.valueOf(s[1]), 
														  curr_name
														);
						
						curr_sourceChannel=s[2];
						
						System.out.println("adding to mapSI_Unique_AuthName_AuthID;curr_name:"+curr_name);
						writerDebug.append("\n adding to mapSI_Unique_AuthName_AuthID;curr_name:"+curr_name+"<->sourceChannel:"+curr_sourceChannel);
						writerDebug.flush();
					}
					if(mapSI_Unique_AuthName_AuthID.size()==0){
						System.out.println("----ERROR:load failed mapSI_Unique_AuthName_AuthID...");
						return false;
					}
			
			// STEP 2 : read unique NOUNS for each doc
			TreeMap<String, String> mapUnique_Nouns_CSV_EachLine=new TreeMap<String, String>();
			try{
				mapUnique_Nouns_CSV_EachLine=
				readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
				readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile(
																	inFile_2_Distinct_Noun_eachLine_as_doc,
																	-1, 
																	-1, 
																	"",//outFile, 
																	false, //is_Append_outFile, 
																	false,//is_Write_To_OutputFile, 
																	"debug_label", //, 
																	-1, // token for  Primary key
																	false // isSOPprint
																);
				
				System.out.println("loaded mapUnique_Nouns_CSV_EachLine" +mapUnique_Nouns_CSV_EachLine.size()
										+" inFile_2_Distinct_Noun_eachLine_as_doc:"+inFile_2_Distinct_Noun_eachLine_as_doc );	
			}
			catch(Exception e){
				System.out.println("ERROR:: step 1...");
				e.printStackTrace();
			}
			if(mapUnique_Nouns_CSV_EachLine.size()==0){
				System.out.println("ERROR:: load failed mapUnique_Nouns_CSV_EachLine...");
				return false;
			}
			
			int linNo_=0; // lineno is not docID
			int docID=0;
			// <verb!#!#noun!#!#docID>
			for(String curr_doc_dist_noun:mapUnique_Nouns_CSV_EachLine.keySet() ){
				linNo_++;
				
				String []s =curr_doc_dist_noun.split("!#!#");
				//NOUN
				String t=s[1].replace("!!!!!!", "!!!").replace("!!!!!!", "!!!").replace("!!!!!!", "!!!")
							.replace("!!!!!!", "!!!").replace("!!!!!!", "!!!").replace("!!!!!!", "!!!").replace("!!!!!!", "!!!");
				
				docID= Integer.valueOf(s[2]);
				//
				map_eachDoc_Unique_Nouns_count.put(docID, new Double(s.length) );
				
				//max docID to know total num of docs
				if(docID > MAX_docID_of_ )
					MAX_docID_of_=docID;
				
			}
			
			
			
			if(map_eachDoc_Unique_Nouns_count.size()==0){
				System.out.println("ERROR: load failed map_eachDoc_Unique_Nouns_count...");
				return false;
			}
			TreeMap<String, String> map_eachDoc_TF_IDF_Feat_Vect=new TreeMap<String, String>();
			// STEP 3: LOAD TF-IDF of each doc (each line is a doc)
			try{
				map_eachDoc_TF_IDF_Feat_Vect=
						readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
						readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile(
													inFile_1_tf_idf_each_doc,
													-1, 
													-1, 
													"",//outFile, 
													false, //is_Append_outFile, 
													false,//is_Write_To_OutputFile, 
													"debug_label", //, 
													-1, // token for  Primary key
													false // isSOPprint
													);
				System.out.println("LOADED map_eachDoc_TF_IDF_Feat_Vect:"+map_eachDoc_TF_IDF_Feat_Vect.size()
									+" from inFile_1_tf_idf_each_doc:"+inFile_1_tf_idf_each_doc);
			}
			catch(Exception e){
				System.out.println("step 1...");
				e.printStackTrace();
			}
			if(map_eachDoc_TF_IDF_Feat_Vect.size()==0){
				System.out.println("ERROR: load failed map_eachDoc_TF_IDF_Feat_Vect...");
				return false;
			}
			int lineNumber=1;
			
			FileWriter writerTFIDF_Avg_Sum=new FileWriter(new File(baseFolder+"tf_idf_sum_avg.txt"));
			
			// each document (each line) sum up all tf-idf and get total, average etc.
			for(String eachLine:map_eachDoc_TF_IDF_Feat_Vect.keySet()){
				if(eachLine.length()<1) continue;
				String [] s2= eachLine.split("!!!");
				
				String [] s=s2[1].split(" ");
				int c=0; double sum_curr_tf_idf=0;
				//System.out.println("l:"+eachLine);
				if(s.length==0) {
					continue;
				}
				//
				while(c<s.length){
					sum_curr_tf_idf=sum_curr_tf_idf+Double.valueOf(s[c].split(":")[1]);
					c++;
				}
				double avg_curr_tf_idf=sum_curr_tf_idf/Double.valueOf(c);
				System.out.println(" c:"+c);
				//doc_id!!!sum_tf_idf!!!avg_tf_idf
				writerTFIDF_Avg_Sum.append(s2[0]+"!!!"+sum_curr_tf_idf+"!!!"+avg_curr_tf_idf+"\n");
				writerTFIDF_Avg_Sum.flush();
				lineNumber++;
			}
			
//			BufferedReader reader_SUM_AVG_TF_IDF=
//								new BufferedReader(new FileReader(new File(baseFolder+"tf_idf_sum_avg.txt")));
			
			//STEP 4: For topics in array given_N_topics, for each author Find count of documents.
			try{

				map_eachLine_as_doc_bodyText=
						readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
						readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
																		inFile_4_contains_only_bodyText,
																		-1, 
																		-1,
																		 "load bodytext",//debug_label,
																		 false // isPrintSOP
																		);
				System.out.println("loaded map_eachLine_as_doc_bodyText:"+map_eachLine_as_doc_bodyText.size()
									+" inFile_4_contains_only_bodyText:"+inFile_4_contains_only_bodyText );
			
			}
			catch(Exception e){
				System.out.println("ERROR: step 1...");
				e.printStackTrace();
			}
			if(map_eachLine_as_doc_bodyText.size()==0){
				System.out.println("ERROR: load failed map_eachLine_as_doc_bodyText...");
				return false;
			}
			
			//STEP 4: For topics in array given_N_topics, for each author Find count of documents.

			try{
			map_eachLine_as_doc_all_tokens=
					readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
					readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
												inFile_3_with_all_tokens,
												-1, 
												-1, 
												 "load bodytext",//debug_label,
												 false // isPrintSOP
												);
			System.out.println("loaded map_eachLine_as_doc_all_tokens:"+map_eachLine_as_doc_all_tokens.size()
								+"inFile_3_with_all_tokens:"+inFile_3_with_all_tokens);	
			}
			catch(Exception e){
				System.out.println("ERROR: load failed map_eachLine_as_doc_all_tokens...");
				e.printStackTrace();
				return false;
			}
			
			// APPROACH 1
			if(Flag_2_choose_approach_topicDoc_count_for_each_author==1){
				
			System.out.println("RUNNING all tokens");
			int total_all_token_size=map_eachLine_as_doc_all_tokens.size();
			//
			for(int lineNo:map_eachLine_as_doc_all_tokens.keySet()){

				String []s=map_eachLine_as_doc_all_tokens.get(lineNo).split("!!!");
				//if(lineNo<50000) { System.out.println("SKIPPING: lineNo:"+lineNo );continue;} //FAST DEBUG
				if(lineNo<=10)
				System.out.println("23.line:"+map_eachLine_as_doc_all_tokens.get(lineNo)
											);
				System.out.println("24. lineNo:"+lineNo +" out of total="+total_all_token_size);
				
				// 
				if(token_no_contain_bodyText_in_inFile_3  > map_eachLine_as_doc_all_tokens.size()
						|| token_no_contain_authors_in_inFile_3  > map_eachLine_as_doc_all_tokens.size()){
					writerDebug.append("\n ERROR : mismatch in tokens: lineNo:"+lineNo+" lineNo:"+lineNo
									
									);
					writerDebug.flush();
					System.out.println("CONTINUE..");
					continue;
				}
				 
				
				try{
				map_eachLine_as_doc_AuthorName_AND_bodyText.put(lineNo, s[token_no_contain_authors_in_inFile_3-1]+" "
															+ s[token_no_contain_bodyText_in_inFile_3-1]
													);
				 
				
				}
				catch(Exception e){
					System.out.println(" error 20 : token ");
					writerDebug.append("\n ERROR: token_no_contain_authors_in_inFile_3:"+
								token_no_contain_authors_in_inFile_3
							+" token_no_contain_bodyText_in_inFile_3:"+token_no_contain_bodyText_in_inFile_3
							+" map_eachLine_as_doc_all_tokens.size:"+map_eachLine_as_doc_all_tokens.size()
							+" error:"+e.getMessage()
							);
					writerDebug.flush();
					continue;
				}
//				System.out.println("concatenate->"+s[token_no_contain_authors_in_inFile_3-1]+" "
//															+ s[token_no_contain_bodyText_in_inFile_3-1]);
			} //END for(int lineNo:map_eachLine_as_doc_all_tokens.keySet()){
			
			if(map_eachLine_as_doc_AuthorName_AND_bodyText.size()==0){
						System.out.println("ERROR: load failed map_eachLine_as_doc_AuthorName_AND_bodyText...");
				return false;
			}
			
			System.out.println("map_eachLine_as_doc_bodyText(size):"+map_eachLine_as_doc_bodyText.size());
			
			int debug_auth_count=0;int auth_id=-1;
			int total_uniq_auth=mapUnique_Authors_Freq.size();
			// iterate each unique authors
			for(String curr_author_name:mapUnique_Authors_Freq.keySet()){
				debug_auth_count++;
				if(debug_auth_count<3000) {
					//System.out.println("SKIPPING debug_auth_count:"+debug_auth_count + " out of total_uniq_auth="+total_uniq_auth);
					//continue; //fast debug
				}
				System.out.println("debug_auth_count:"+debug_auth_count + " out of total_uniq_auth="+total_uniq_auth);
				curr_author_name=p6_MAIN_Work.p6_normalize_auth_name(curr_author_name).replace("auth:from: ", "");
				int debug_currLine_cnt=0;
				int cnt=0;
				TreeMap<String, Integer> map_QueryTopic_Freq=new TreeMap<String, Integer>();
				
				
				// APPROACH 1: based on Author Name (not so perfect)
				 
				 //each query topic, get distinct count of documents
				while(cnt<given_N_topics.length){
					auth_id=-1;
					String curr_n_topic=given_N_topics[cnt];
					//iterate over AuthorNAME and BODYTEXT to match for author Name and N_topics
					for(int currNo:map_eachLine_as_doc_AuthorName_AND_bodyText.keySet()){
						debug_currLine_cnt++;
//						System.out.println("running curr_n_topic:"+curr_n_topic+" cnt:"+cnt+" debug_currLine_cnt:"
//										   +debug_currLine_cnt+" debug_auth_count:"+debug_auth_count);
//						String currLine_of_all_token=map_eachLine_as_doc_all_tokens.get(currNo);
//						String currLine_of_bodyText=map_eachLine_as_doc_bodyText.get(currNo);
						String currLine_of_authName_and_bodyText=
														map_eachLine_as_doc_AuthorName_AND_bodyText.get(currNo);
						
						int is_auth_present=currLine_of_authName_and_bodyText.indexOf(curr_author_name);
						int is_q_Topic_present=currLine_of_authName_and_bodyText.indexOf(curr_n_topic);
 
						//auth id=
						if(mapSI_Unique_AuthName_AuthID.containsKey(curr_author_name)){
							auth_id=mapSI_Unique_AuthName_AuthID.get(curr_author_name);
							//System.out.println("AUTH ID FOUND:"+currNo+" name:"+curr_author_name);
						}
						else{
//							System.out.println("AUTH ID not found in mapSI_Unique_AuthName_AuthID:"+currNo
//												+" name:"+curr_author_name);
						}
						// Nth topic matches AND author_name matches
						if(is_auth_present>=0 &&
							is_q_Topic_present >=0){

//							System.out.println("1.curr_author_name:"+curr_author_name);
//							System.out.println("map_authorName_N_topic_freq:"+map_authorName_N_topic_freq);
//							System.out.println("mapSI_Unique_AuthName_AuthID:"+mapSI_Unique_AuthName_AuthID);
							//Some author names not available author name in global?
							if(!mapSI_Unique_AuthName_AuthID.containsKey(curr_author_name)){
								writerDebug.append("\n not available author name in global: "+curr_author_name);
								writerDebug.flush();
							}
							else if(!map_authorName_N_topic_freq.containsKey(curr_author_name)){
								map_QueryTopic_Freq.put(curr_n_topic, 1);
								map_authorName_N_topic_freq.put(curr_author_name,map_QueryTopic_Freq); //authName as key
//								System.out.println("debug:mapSI_Unique_AuthName_AuthID.get(curr_author_name):"
//												+mapSI_Unique_AuthName_AuthID+" curr_author_name:"+curr_author_name
//												+"contains:"+mapSI_Unique_AuthName_AuthID.containsKey(curr_author_name);

								map_authID_N_QueryTopic_freq.put(mapSI_Unique_AuthName_AuthID.get(curr_author_name)
																,map_QueryTopic_Freq); //authID as key
							}
							else{
								
								TreeMap<String, Integer> tempMap=map_authorName_N_topic_freq.get(curr_author_name);
								// if Q topic exists for this author ID 
								if(map_QueryTopic_Freq.containsKey(curr_n_topic)){
									int new_freq=map_QueryTopic_Freq.get(curr_n_topic) +1;
									tempMap.put(curr_n_topic, new_freq);
									map_authorName_N_topic_freq.put(curr_author_name, tempMap); //authName as Key
									map_authID_N_QueryTopic_freq.put(mapSI_Unique_AuthName_AuthID.get(curr_author_name)
																	, tempMap); //authID as key
								}
								else{
									tempMap.put(curr_n_topic, 1);
									map_authorName_N_topic_freq.put(curr_author_name, tempMap); //authName as Key
									map_authID_N_QueryTopic_freq.put(mapSI_Unique_AuthName_AuthID.get(curr_author_name)
																	, tempMap); //authID as key
									
								}
								
							}
						//	System.out.println("******PRESENT: AUTHOR and TOPIC->debug_auth_count:"+debug_auth_count);
						}
						else{
//							System.out.println("NOT PRESENT: AUTHOR and TOPIC->"+" debug_auth_count="+debug_auth_count
//												+" is_auth_present:"+is_auth_present+" is_q_Topic_present:"+is_q_Topic_present);
						}
					}
					cnt++;
				} //end of given_N_topics
				
				
				// debug
				if(map_authID_N_QueryTopic_freq.containsKey(auth_id)){
					writerDebug.append("\n curr_author_name:"+curr_author_name+"!!!"+
										map_authID_N_QueryTopic_freq.get(auth_id)+"!!!"
										+map_authID_N_QueryTopic_freq.size()+"!!!"+map_authorName_N_topic_freq.size()+
										"\n");
					
					writerDebug.flush();
				}

			} // END iterate each unique authors
			 
			} //if(Flag_2_choose_approach_topicDoc_count_for_each_author==1){
			
			TreeMap<Integer, String> map_eachLine_auth_id_doc_id_queryTopicRelated=new TreeMap<Integer, String>();
			// Flag_2_choose_approach_topicDoc_count_for_each_author
			if(Flag_2_choose_approach_topicDoc_count_for_each_author==2){
				System.out.println(" inside Flag_2_choose_approach_topicDoc_count_for_each_author=2");
				try{
					
					map_eachLine_auth_id_doc_id_queryTopicRelated=
							readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
							readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
																			inFile_7_auth_id_doc_id_queryTopicRelated,
																			-1, 
																			-1,
																			 "load approach2",//debug_label,
																			 false // isPrintSOP
																			);
					
					System.out.println("LOADED map_eachLine_auth_id_doc_id_queryTopicRelated.size:"
										+map_eachLine_auth_id_doc_id_queryTopicRelated.size());
					
					writerDebug.append("\n LOADED map_eachLine_auth_id_doc_id_queryTopicRelated.size:"
										+map_eachLine_auth_id_doc_id_queryTopicRelated.size());
					writerDebug.flush();
					
					// 
					for(int seq_: map_eachLine_auth_id_doc_id_queryTopicRelated.keySet()){
						
						String currLine=map_eachLine_auth_id_doc_id_queryTopicRelated.get(seq_);
						

						if(seq_ <10)
							System.out.println( " currLine:"+currLine);
						
						String []s =currLine.split("!!!"); //<authID!!!docID!!!relatedQUERY>
						int curr_auth_id= Integer.valueOf(s[0]);
						String curr_query_topic=s[2];
						
//						writerDebug.append("\n seq_:"+seq_ + " curr_auth_id:"+ curr_auth_id+" curr_query_topic:"+curr_query_topic );
						writerDebug.flush();
						
						if(seq_ <10)
							System.out.println("curr_auth_id:"+ curr_auth_id+" curr_query_topic:"+curr_query_topic+" currLine:"+currLine);						
						else{
							if(seq_%10000==0)
								System.out.println("lineNo->"+seq_);
						}
						
//						if(!mapSI_Unique_AuthName_AuthID.containsValue(curr_auth_id)){
//							writerDebug.append("\n not available author name in global: curr_auth_id"+curr_auth_id);
//							writerDebug.flush();
//						}
//						else
						if(!map_authID_N_QueryTopic_freq.containsKey(curr_auth_id)){
							TreeMap<String, Integer> tempMap=new TreeMap<String, Integer>();
							tempMap.put(curr_query_topic, 1); 
//							System.out.println("debug:mapSI_Unique_AuthName_AuthID.get(curr_author_name):"
//											+mapSI_Unique_AuthName_AuthID+" curr_author_name:"+curr_author_name
//											+"contains:"+mapSI_Unique_AuthName_AuthID.containsKey(curr_author_name);

							map_authID_N_QueryTopic_freq.put(curr_auth_id
															,tempMap); //authID as key
						}
						else{
							
							TreeMap<String, Integer> tempMap=map_authID_N_QueryTopic_freq.get(curr_auth_id);
							// if Q topic exists for this author ID 
							if(tempMap.containsKey(curr_query_topic)){
								int new_freq=tempMap.get(curr_query_topic) +1;
								tempMap.put(curr_query_topic, new_freq);
//								map_authorName_N_topic_freq.put(curr_author_name, tempMap); //authName as Key
								map_authID_N_QueryTopic_freq.put(curr_auth_id
																, tempMap); //authID as key
							}
							else{
								tempMap.put(curr_query_topic, 1);
//								map_authorName_N_topic_freq.put(curr_author_name, tempMap); //authName as Key
								map_authID_N_QueryTopic_freq.put(curr_auth_id
																, tempMap); //authID as key
								
							}
							
						} //END if(!mapSI_Unique_AuthName_AuthID.containsValue(curr_auth_id)){
						
						writerDebug.append("map_authID_N_QueryTopic_freq.size:"+map_authID_N_QueryTopic_freq.size()+"\n");
						writerDebug.flush();
					}
				
				}
				catch(Exception e){
					System.out.println("ERROR here:"+e.getMessage());
					writerDebug.append("ERROR here:"+e.getMessage()+"\n");
					writerDebug.flush();
					e.printStackTrace();
					
				}
			}

			//first iterate and get only MIN authID 
			for(int authID:map_authID_N_QueryTopic_freq.keySet()){
				System.out.println("map_authID_minAuthId_inFile10.size:"+map_authID_minAuthId_inFile10.size() +" "+authID+" "
												+map_authID_minAuthId_inFile10.containsKey(authID));
				if(map_authID_minAuthId_inFile10.containsKey(authID)){
					int MIN_authID=map_authID_minAuthId_inFile10.get(authID);
					if(authID == MIN_authID){ // ORIGINAL (MIN) AUTHNAME
						map_authID_N_QueryTopic_freq_noDupApplyingMINauthID.put(MIN_authID, map_authID_N_QueryTopic_freq.get(authID));
					}
				}
			}
			// 
			writerDebug.append("Total COUNT OF min AUTH ID:"+map_authID_N_QueryTopic_freq_noDupApplyingMINauthID.size()+"\n");
			writerDebug.flush();
			
			// map_authID_N_QueryTopic_freq --- remove duplicates in AuthNAME using MIN authID 
			for(int authID:map_authID_N_QueryTopic_freq.keySet()){
				if(!map_authID_minAuthId_inFile10.containsKey(authID)){
					writerDebug.append("NOT FOUND authID:"+authID+"\n"); writerDebug.flush();
					continue;
				}
				
				int MIN_authID=map_authID_minAuthId_inFile10.get(authID);
				
				// DUPLICATE authNAME, MERGE frequency numbers to MIN authID
				if(authID != MIN_authID){
					if(map_authID_N_QueryTopic_freq_noDupApplyingMINauthID.containsKey(MIN_authID)==false 
							|| map_authID_N_QueryTopic_freq.containsKey(authID)==false ){
						continue;}
					
					//THIS should already exist as ABOVE for iteration
					TreeMap<String,Integer> map_currQueryFreq_MIN=map_authID_N_QueryTopic_freq_noDupApplyingMINauthID.get(MIN_authID);
					TreeMap<String,Integer> map_currQueryFreq_NOT_MIN=map_authID_N_QueryTopic_freq.get(authID);
					
					writerDebug.append( "authID != MIN_authID:("+authID+"!="+ MIN_authID+")"+
												map_currQueryFreq_MIN+" NOT:"+map_currQueryFreq_NOT_MIN+"\n");
					writerDebug.flush();
					
					// each of query from MIN AUTHID
					for(String curr_query:map_currQueryFreq_NOT_MIN.keySet()){
						
						if(map_currQueryFreq_MIN.containsKey(curr_query)){// COMMON query exists
							int new_freq=map_currQueryFreq_MIN.get(curr_query)+ map_currQueryFreq_NOT_MIN.get(curr_query);
							map_currQueryFreq_MIN.put(curr_query, new_freq);
						}
						else{
							map_currQueryFreq_MIN.put(curr_query, map_currQueryFreq_NOT_MIN.get(curr_query));
						}
						
					}
					//merged 
					map_authID_N_QueryTopic_freq_noDupApplyingMINauthID.put(MIN_authID, map_currQueryFreq_MIN);
					writerDebug.append(" merged:"+map_authID_N_QueryTopic_freq_noDupApplyingMINauthID.get(MIN_authID)+"\n");
					writerDebug.flush();
				}
			}
			 
			if(map_authorName_N_topic_freq.size()==0&&  map_authID_N_QueryTopic_freq.size()==0){
				System.out.println("load failed map_authorName_N_topic_freq..."
								+" map_eachLine_as_doc_AuthorName_AND_bodyText:"+map_eachLine_as_doc_AuthorName_AND_bodyText.size()
								);
				System.out.println("FAILED map_authID_N_QueryTopic_freq:"+map_authID_N_QueryTopic_freq.size());
				return false;	
			}
			
			
			// STEP 4.1: Load Person , Organization, Location , Number etc.
			 TreeMap<String, String>
			  map_DocID_Person=LoadMultipleValueToMap2_AS_KEY_VALUE.LoadMultipleValueToMap2_AS_KEY_VALUE(
					  																							inFile_8_get_organization_person_location_docID,
															  													"@@@@",
															  													"6,2",
				  																								"", //append_suffix_at_each_line
															  													false,
															  													false //isPrintSOP
															  													 );
			 System.out.println("Loaded docid, organization: map_DocID_Person.size:"+map_DocID_Person.size());
			 
			 writerDebug.append("Loaded docid, organization: map_DocID_Person.size:"+map_DocID_Person.size() +"\n");
			 writerDebug.flush();
			
			 TreeMap<String, String>
			  map_DocID_Organization=LoadMultipleValueToMap2_AS_KEY_VALUE.LoadMultipleValueToMap2_AS_KEY_VALUE(
					  																							inFile_8_get_organization_person_location_docID,
															  													"@@@@",
															  													"6,3",
				  																								"", //append_suffix_at_each_line
															  													false,
															  													false //isPrintSOP
															  													 );
			
			 System.out.println("Loaded  map_DocID_Organization.size:"+map_DocID_Organization.size());
			 writerDebug.append("Loaded  map_DocID_Organization.size:"+map_DocID_Organization.size()+"\n");
			 writerDebug.flush();
			 
			 TreeMap<String, String>
			  map_DocID_Location=LoadMultipleValueToMap2_AS_KEY_VALUE.LoadMultipleValueToMap2_AS_KEY_VALUE(
					  																							inFile_8_get_organization_person_location_docID,
															  													"@@@@",
															  													"6,4",
				  																								"", //append_suffix_at_each_line
															  													false,
															  													false //isPrintSOP
															  													 );
			 
			 System.out.println("Loaded  map_DocID_Location.size:"+map_DocID_Location.size());
			 writerDebug.append("Loaded  map_DocID_Location.size:"+map_DocID_Location.size() +"\n");
			 writerDebug.flush();
			 
			 writerDebug.append( " map_DocID_Location:" +map_DocID_Location+"\n");
			 writerDebug.flush();
			 
			 TreeMap<String, String>
			  map_DocID_Numbers=LoadMultipleValueToMap2_AS_KEY_VALUE.LoadMultipleValueToMap2_AS_KEY_VALUE(
					  																							inFile_8_get_organization_person_location_docID,
															  													"@@@@",
															  													"6,5",
				  																								"", //append_suffix_at_each_line
															  													false,
															  													false //isPrintSOP
															  													 );
			 
			 System.out.println("Loaded  map_DocID_Numbers.size:"+map_DocID_Numbers.size());
			 writerDebug.append("Loaded  map_DocID_Numbers.size:"+map_DocID_Numbers.size() +"\n");
			 writerDebug.flush();
			 
			 
			
			//STEP 5: READ inFile_5_AuthorID_Doc_ID_matrix, << FOR EACH AUTHOR , CALC METRICS >> 
			//Metrics: Cummulative TF-IDF, COUNT(DISTINCT nouns), Distint(Docs), DISTINCT(Docs=Query) <--Query={China,India, Shark attack etc}
			TreeMap<String, String> map_eachLine_AuthID_DocID_matrix_row=new TreeMap<String, String>();
			TreeMap<Integer, Double> map_DocID_SUM_TF_IDF = new TreeMap<Integer, Double>();
			TreeMap<Integer, Double> map_DocID_AVG_TF_IDF = new TreeMap<Integer, Double>();
			try{
			map_eachLine_AuthID_DocID_matrix_row=
					readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
					readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile(
												inFile_5_AuthorID_Doc_ID_matrix,
												-1, 
												-1, 
												"",//outFile, 
												false, //is_Append_outFile, 
												false,//is_Write_To_OutputFile, 
												"debug_label", //, 
												-1, // token for  Primary key
												false // isSOPprint
												);
			
			System.out.println("loaded map_eachLine_AuthID_DocID_matrix_row:"+map_eachLine_AuthID_DocID_matrix_row.size()
							+" inFile_5_AuthorID_Doc_ID_matrix:"+inFile_5_AuthorID_Doc_ID_matrix);
			}
			catch(Exception e){
				System.out.println("step 1...");
				e.printStackTrace();
			}
			if(map_eachLine_AuthID_DocID_matrix_row.size()==0){
				System.out.println("load failed map_eachLine_AuthID_DocID_matrix_row...");
				return false;	
			}
			// Load SUM-AVG TF-IDF of each document
			TreeMap<Integer, String> map_eachLine_as_doc_SUM_AVG_TF_IDF=new TreeMap<Integer, String>();
			try{
				map_eachLine_as_doc_SUM_AVG_TF_IDF=
					readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
					readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
															baseFolder+"tf_idf_sum_avg.txt",
															-1, 
															-1,
															"debug_label",
															true //isSOPprint
															);
				
				System.out.println("loaded map_eachLine_as_doc_SUM_AVG_TF_IDF:"+map_eachLine_as_doc_SUM_AVG_TF_IDF.size()
									+" SUM_AVG_FILE:"+baseFolder+"tf_idf_sum_avg.txt");
			}
			catch(Exception e){
				System.out.println("step 1...");
				e.printStackTrace();
			}
			if(map_eachLine_as_doc_SUM_AVG_TF_IDF.size()==0){
				System.out.println("load failed map_eachLine_as_doc_SUM_AVG_TF_IDF...");
				return false;	
			}
			

			writerDebug.append("\nmap_eachLine_as_doc_SUM_AVG_TF_IDF(size):"+map_eachLine_as_doc_SUM_AVG_TF_IDF.size()+"\n");
			writerDebug.flush();
			// separate map for SUM and AVG OF TF-IDF
			for(int lineno_curr_doc_id_sum_avg_tf_idf:map_eachLine_as_doc_SUM_AVG_TF_IDF.keySet()){
				System.out.println("lineno_curr_doc_id_sum_avg_tf_idf:"+lineno_curr_doc_id_sum_avg_tf_idf);
				String doc_id_SUM_AVG_tf_idf=
					map_eachLine_as_doc_SUM_AVG_TF_IDF.get(lineno_curr_doc_id_sum_avg_tf_idf);
					String [] metric_tf_idf=doc_id_SUM_AVG_tf_idf.split("!!!");
					int curr_docID=new Integer(metric_tf_idf[0]);
					// SUM
					map_DocID_SUM_TF_IDF.put(curr_docID, Double.valueOf(metric_tf_idf[1]));
					//AVG
					map_DocID_AVG_TF_IDF.put(curr_docID,  Double.valueOf(metric_tf_idf[2]));
					
			}
			if(map_DocID_SUM_TF_IDF.size()==0){
				System.out.println("load failed map_DocID_SUM_TF_IDF...");
				return false;	
			}
			
			FileWriter writer_each_Auth_ID_CUMM_Avg_Sum=new FileWriter(new File(output_Ground_Truth));
			FileWriter writer_each_Auth_ID_CUMM_Avg_Sum_noDup=new FileWriter(new File(output_Ground_Truth_removeDuplicate));
			
			writer_each_Auth_ID_CUMM_Avg_Sum.append("auth_ID!!!Total_docs!!!SUM_tf_idf!!!AVG_tf_idf!!!avgNOUNS!!!tNOUNS!!!Q_topic_Freq!!!Author_Name"
													 +"!!!curr_auth_cumm_SUM_person"
													 +"!!!curr_auth_cumm_AVG_person"
													 +"!!!curr_auth_cumm_SUM_organization"
													 +"!!!curr_auth_cumm_AVG_organization"
													 +"!!!curr_auth_cumm_SUM_location"
													 +"!!!curr_auth_cumm_AVG_location"
													 +"!!!curr_auth_cumm_SUM_numbers"
													 +"!!!curr_auth_cumm_AVG_numbers"
													 + "\n");
			writer_each_Auth_ID_CUMM_Avg_Sum_noDup.append("auth_ID!!!Total_docs!!!SUM_tf_idf!!!AVG_tf_idf!!!avgNOUNS!!!tNOUNS!!!Q_topic_Freq!!!Author_Name"
														 +"!!!curr_auth_cumm_SUM_person"
														 +"!!!curr_auth_cumm_AVG_person"
														 +"!!!curr_auth_cumm_SUM_organization"
														 +"!!!curr_auth_cumm_AVG_organization"
														 +"!!!curr_auth_cumm_SUM_location"
														 +"!!!curr_auth_cumm_AVG_location"
														 +"!!!curr_auth_cumm_SUM_numbers"
														 +"!!!curr_auth_cumm_AVG_numbers"
														 + "\n");
			writer_each_Auth_ID_CUMM_Avg_Sum_noDup.flush();
			writer_each_Auth_ID_CUMM_Avg_Sum.flush();
			
			writerDebug.append("\nmap_eachLine_AuthID_DocID_matrix_row.size():"+map_eachLine_AuthID_DocID_matrix_row.size());
			writerDebug.flush();
			int total_doc=0; TreeMap<Integer, Integer> map_authID_totalDocs= new TreeMap<Integer, Integer>();
			// total number of documents for each author
			for(int authID:map_authID_N_QueryTopic_freq.keySet()){
				TreeMap<String, Integer> tmpMap=  map_authID_N_QueryTopic_freq.get(authID);
				total_doc=0;
				for(String c_query:tmpMap.keySet()){
					total_doc=total_doc+tmpMap.get(c_query);
				}
				map_authID_totalDocs.put(authID, total_doc);
				writerDebug.append("\n authID and totalDocs:"+authID+" "+total_doc);
				writerDebug.flush();
			}
			writerDebug.append("\n map_authID_totalDocs.size():"+map_authID_totalDocs.size()
								+" ->map_authID_N_QueryTopic_freq.size:"+map_authID_N_QueryTopic_freq.size());
			writerDebug.flush();
			if(map_authID_totalDocs.size()==0){
				System.out.println("load failed map_authID_totalDocs...");
				return false;	
			}
			
			// read Auth_ID and Doc_ID matrix row ; Each Auth_ID 
			for(String eachLine:map_eachLine_AuthID_DocID_matrix_row.keySet()){
				if(eachLine.length()<2){
					System.out.println(" blank line:"+eachLine);
					continue; //blank
				}
//				System.out.println("mat.eachLine:"+eachLine);
				String []s=eachLine.split("!!!");
				int int_curr_auth_ID=Integer.valueOf(s[1]); //first column [0] is blank
//				System.out.println("int_curr_auth_ID:"+int_curr_auth_ID);
				String [] curr_arr_Doc_IDs=s[2].split(",");
				
				int c=0; 
				double curr_auth_cumm_SUM_tf_idf=0.0; //cummulative tf-idf of curr_auth
				double curr_auth_cumm_AVG_tf_idf=0.0;
				
				double curr_auth_cumm_SUM_person=0.0;
				double curr_auth_cumm_AVG_person=0.0;
				double curr_auth_cumm_SUM_organization=0.0;
				double curr_auth_cumm_AVG_organization=0.0;
				double curr_auth_cumm_SUM_location=0.0;
				double curr_auth_cumm_AVG_location=0.0;
				double curr_auth_cumm_SUM_numbers=0.0;
				double curr_auth_cumm_AVG_numbers=0.0;
				
				double curr_auth_cumm_count_NOUNS=0.;
				//
				while(c<curr_arr_Doc_IDs.length){
					String curr_docID = curr_arr_Doc_IDs[c];
					
					//System.out.println("c:"+c);
					double a=0;
					try{
						a=map_DocID_SUM_TF_IDF.get(Integer.valueOf(curr_arr_Doc_IDs[c]));}
					catch(Exception e){
						a=0.;
					}
					double b=0;
					try{
						b=map_DocID_AVG_TF_IDF.get(Integer.valueOf(curr_arr_Doc_IDs[c]));}
					catch(Exception e){
						b=0.;
					}
					double a1=0; //ORGANIZATION
					try{
						a1=Integer.valueOf( map_DocID_Organization.get(curr_docID).split(",").length ); 
					}
					catch(Exception e){
						a1=0.;
					}
					double a2=0; //PERSON
					try{
						a2=Integer.valueOf( map_DocID_Person.get(curr_docID).split(",").length ); 
					}
					catch(Exception e){
						a2=0.;
					}
					double a3=0; //LOCATION
					try{
						a3=Integer.valueOf( map_DocID_Location.get(curr_docID).split(",").length ); 
					}
					catch(Exception e){
						a3=0.;
					}
					double a4=0; //LOCATION
					try{
						a4=Integer.valueOf( map_DocID_Numbers.get(curr_docID).split(",").length ); 
					}
					catch(Exception e){
						a4=0.;
					}
					
					curr_auth_cumm_SUM_tf_idf=curr_auth_cumm_SUM_tf_idf+a;
					curr_auth_cumm_AVG_tf_idf=curr_auth_cumm_AVG_tf_idf+b;
					
					curr_auth_cumm_SUM_organization=curr_auth_cumm_SUM_organization+a1;
					curr_auth_cumm_SUM_person=curr_auth_cumm_SUM_person +a2;
					curr_auth_cumm_SUM_location=curr_auth_cumm_SUM_location +a3;
					curr_auth_cumm_SUM_numbers=curr_auth_cumm_SUM_numbers +a4;
					
					
					System.out.println("doc id:"+Integer.valueOf(curr_arr_Doc_IDs[c]));
					//NOUN 
					//System.out.println("map_eachDoc_Unique_Nouns_count:"+map_eachDoc_Unique_Nouns_count);
					double curr_f=0.;
					try{
						curr_f= map_eachDoc_Unique_Nouns_count.get(Integer.valueOf(curr_arr_Doc_IDs[c]));
					}
					catch(Exception e){
						curr_f=0.;
					}
					
					// for all documents of curr_auth_id, find total NOUN
					curr_auth_cumm_count_NOUNS=curr_auth_cumm_count_NOUNS+curr_f;
					
					c++;
				}
				System.out.println("int_curr_auth_ID:"+int_curr_auth_ID);
				
				writerDebug.append("\n out->"+int_curr_auth_ID+"!!!"+curr_auth_cumm_SUM_tf_idf
														+"!!!"+curr_auth_cumm_AVG_tf_idf
														+"!!!"+curr_auth_cumm_count_NOUNS
														+"!!!"+map_authID_N_QueryTopic_freq.containsKey(int_curr_auth_ID)
														+"!!!"+mapSI_Unique_AuthID_AuthName.containsKey(int_curr_auth_ID)
														);
				writerDebug.append("\n out1->"+map_authID_N_QueryTopic_freq.size());
				writerDebug.append("\n out2->"+mapSI_Unique_AuthName_AuthID.size());
				writerDebug.flush();
				double curr_avg_nouns=0.0;
				try{
					curr_avg_nouns=curr_auth_cumm_count_NOUNS /new Double(map_authID_totalDocs.get(int_curr_auth_ID)); //avgNOUNS
				}
				catch(Exception e ){
					curr_avg_nouns=00.00;
				}
				try{ 
					curr_auth_cumm_AVG_organization=curr_auth_cumm_SUM_organization/new Double(map_authID_totalDocs.get(int_curr_auth_ID));
				}
				catch(Exception e ){
					curr_auth_cumm_AVG_organization=0.0;
				}
				try{
					curr_auth_cumm_AVG_person=curr_auth_cumm_SUM_person/new Double(map_authID_totalDocs.get(int_curr_auth_ID));
				}
				catch(Exception e ){
					curr_auth_cumm_AVG_person=0.0;
				}
				try{
					curr_auth_cumm_AVG_location=curr_auth_cumm_SUM_location/new Double(map_authID_totalDocs.get(int_curr_auth_ID));
				}
				catch(Exception e ){
					curr_auth_cumm_AVG_location=0.0;
				}
				try{
					curr_auth_cumm_AVG_numbers=curr_auth_cumm_SUM_numbers/new Double(map_authID_totalDocs.get(int_curr_auth_ID));
				}
				catch(Exception e ){
					curr_auth_cumm_AVG_numbers=0.0;
				}
				
				//int int_curr_auth_curr_doc_ids=Integer.valueOf(curr_arr_Doc_IDs[c]); 
				writer_each_Auth_ID_CUMM_Avg_Sum.append( int_curr_auth_ID
														+"!!!"+map_authID_totalDocs.get(int_curr_auth_ID)
														+"!!!"+curr_auth_cumm_SUM_tf_idf
														+"!!!"+curr_auth_cumm_AVG_tf_idf
														+"!!!"+curr_avg_nouns
														+"!!!"+curr_auth_cumm_count_NOUNS
														+"!!!"+map_authID_N_QueryTopic_freq.get(int_curr_auth_ID)
														+"!!!"+mapSI_Unique_AuthID_AuthName.get(int_curr_auth_ID)
														+"!!!"+curr_auth_cumm_SUM_person
														+"!!!"+curr_auth_cumm_AVG_person
														+"!!!"+curr_auth_cumm_SUM_organization
														+"!!!"+curr_auth_cumm_AVG_organization
														+"!!!"+curr_auth_cumm_SUM_location
														+"!!!"+curr_auth_cumm_AVG_location
														+"!!!"+curr_auth_cumm_SUM_numbers
														+"!!!"+curr_auth_cumm_AVG_numbers
														+"\n");
				writer_each_Auth_ID_CUMM_Avg_Sum.flush();
				
				if(map_authID_N_QueryTopic_freq_noDupApplyingMINauthID.containsKey(int_curr_auth_ID)){
					//
					writer_each_Auth_ID_CUMM_Avg_Sum_noDup.append( int_curr_auth_ID
														+"!!!"+map_authID_totalDocs.get(int_curr_auth_ID)
														+"!!!"+curr_auth_cumm_SUM_tf_idf
														+"!!!"+curr_auth_cumm_AVG_tf_idf
														+"!!!"+curr_avg_nouns
														+"!!!"+curr_auth_cumm_count_NOUNS
														+"!!!"+map_authID_N_QueryTopic_freq_noDupApplyingMINauthID.get(int_curr_auth_ID)
														+"!!!"+mapSI_Unique_AuthID_AuthName.get(int_curr_auth_ID)
														+"!!!"+curr_auth_cumm_SUM_person
														+"!!!"+curr_auth_cumm_AVG_person
														+"!!!"+curr_auth_cumm_SUM_organization
														+"!!!"+curr_auth_cumm_AVG_organization
														+"!!!"+curr_auth_cumm_SUM_location
														+"!!!"+curr_auth_cumm_AVG_location
														+"!!!"+curr_auth_cumm_SUM_numbers
														+"!!!"+curr_auth_cumm_AVG_numbers
														+"\n");
					writer_each_Auth_ID_CUMM_Avg_Sum_noDup.flush();
				}
				
				
				
				
				
			} //END read Auth_ID and Doc_ID matrix row ; Each Auth_ID
			
			
			System.out.println("----------------------------------");
			System.out.println("mapUnique_Authors_Freq(size):"+mapUnique_Authors_Freq.size());
			System.out.println("mapUnique_Authors_Freq_EachLine(size):"+mapUnique_Authors_Freq_EachLine.size());
			System.out.println("map_authorName_N_topic_freq(size):"+map_authorName_N_topic_freq.size());
			System.out.println("out file:"+baseFolder+"Ground_Truth_OUT_auth_id_CUMM_SUM_AVG_tf_idf.txt");
			System.out.println("-------------------------IMPORTANT----------");
			System.out.println("Size of below 3 maps should be same..");
			System.out.println("map_eachDoc_Unique_Nouns_count(size):"+MAX_docID_of_);
			System.out.println("map_eachLine_as_doc_bodyText(size):"+map_eachLine_as_doc_bodyText.size());
			System.out.println("map_eachLine_as_doc_all_tokens(size):"+map_eachLine_as_doc_all_tokens.size());
			System.out.println("map_authID_totalDocs(size):"+map_authID_totalDocs.size());
			System.out.println("map_authID_N_QueryTopic_freq(size):"+map_authID_N_QueryTopic_freq.size());
			
			writerDebug.append("\nmap_authID_totalDocs:"+map_authID_totalDocs.size());
			writerDebug.append("\nmap_authID_N_QueryTopic_freq:"+map_authID_N_QueryTopic_freq.size());
			writerDebug.flush();
			////
			if(MAX_docID_of_!=map_eachLine_as_doc_bodyText.size()
					|| MAX_docID_of_!=map_eachLine_as_doc_all_tokens.size()
					|| map_eachLine_as_doc_bodyText.size()!=map_eachLine_as_doc_all_tokens.size()
					){
				System.out.println("***ERROR****Size of below 3 maps should be same..");
				System.out.println("map_eachDoc_Unique_Nouns_count(size):MAX_docID_of_"+MAX_docID_of_+" map_eachDoc_Unique_Nouns_count.size:"+map_eachDoc_Unique_Nouns_count.size());
				System.out.println("map_eachLine_as_doc_all_tokens(size):"+map_eachLine_as_doc_all_tokens.size());
				System.out.println("map_eachLine_as_doc_bodyText(size):"+map_eachLine_as_doc_bodyText.size()+"<-obsolete");
				System.out.println("map_authID_totalDocs(size):"+map_authID_totalDocs.size());
			}
			
			
			
		}
		catch(Exception e){
			e.printStackTrace();
		}
		return true;
	}
	// convert map String to TreeMap
	public static TreeMap<String, Integer> convert_mapString_to_TreeMap(String mapString){
		
		TreeMap<String, Integer> map_each_q_topic_freq=new TreeMap<String, Integer>();
		mapString=mapString.replace(", ", ",").replace("{", "").replace("}", "");
		String [] s=mapString.split(",");
		int c=0;
		//
		while(c<s.length){
			if(s[c].indexOf("null")>=0) {c++; continue;}
			System.out.println("s[c]:"+s[c]);
			String s2[]=s[c].split("=");
			map_each_q_topic_freq.put(s2[0] , Integer.valueOf(s2[1]));
			c++;
		}
		return map_each_q_topic_freq;
	}
	// final ground_truth
	public static boolean calc_ground_truth_2( 
											String  baseFolder,
											String  inFile_output_Ground_Truth,
											String  inFile_6_AuthorName_AND_Auth_ID,
											String  inFile_9_auth_id_doc_id_queryTopicRelated__applying_multi_crossAuthIDs,
											String  inFile_10_authName_AND_auth_id__MAPPEDmultiAuthID_SORTED_add_minAuthID,
											String  inFile_11_authName_AND_auth_id_txt_YES_dirtyAuthNames,
											String  inFile_12_authName_AND_auth_id_txt_NO_dirtyAuthNames,
											String  inFile_outputFileName_inFile_3_keyword_AND_freq,
											int     token_containing_mapString_each_Q_Topic_Freq,
											String  output_Ground_Truth_final_1,
											String  output_Ground_Truth_final_2,
											String  output_Ground_Truth_final_3,
											String  output_Ground_Truth_final_4,
											String  output_Ground_Truth_final_5,
											String  output_Ground_Truth_final_6,
											String  output_Ground_Truth_final_7,
											String  output_Ground_Truth_final_8,
											String  output_Ground_Truth_auth_name_freq_final_5,
											int    	N_Top_authors,
											String  pattern_for_sourceChannel_TRAD, 
											String  pattern_for_sourceChannel_POLI, 
											boolean isSOPprint
											){
		String line="";
		TreeMap<String, Integer> map_each_q_topic_freq=new TreeMap<String, Integer>();
		TreeMap<String, Integer> map_authName_AND_freq=new TreeMap<String, Integer>();
		TreeMap<Integer, Integer> map_authID_AND_TotalDocs=new TreeMap<Integer, Integer>();
		
		TreeMap<Integer, Integer> map_authID_minAuthId=new TreeMap<Integer, Integer>();
		TreeMap<Integer, String> map_authID_SourceChannel=new TreeMap<Integer, String>();
		
		
		int flag_approach=2;
		TreeMap<Integer,String> mapAuthId_authName=new TreeMap<Integer, String>();
		TreeMap<Integer,String> map_YES_DIRTY_AuthId_authName=new TreeMap<Integer, String>();
		try{
			//<authName>!!!<authID>!!!<sourceChannel>
			TreeMap<Integer, String> map_eachLine_authName_AND_auth_id_txt_YES_dirtyAuthNames=
			readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
			readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
																			inFile_11_authName_AND_auth_id_txt_YES_dirtyAuthNames,
																			-1, 
																			-1, 
																			 "load bodytext",//debug_label,
																			 false // isPrintSOP
																			);
			System.out.println( "map_eachLine_authName_AND_auth_id_txt_YES_dirtyAuthNames:"+map_eachLine_authName_AND_auth_id_txt_YES_dirtyAuthNames.size() );
			//<authName>!!!<authID>!!!<sourceChannel>
			for(int seq:map_eachLine_authName_AND_auth_id_txt_YES_dirtyAuthNames.keySet()){
				String currLine=map_eachLine_authName_AND_auth_id_txt_YES_dirtyAuthNames.get(seq);
				String []s=currLine.split("!!!");
				map_YES_DIRTY_AuthId_authName.put(Integer.valueOf(s[1]) ,s[0]);
				map_authID_SourceChannel.put(Integer.valueOf(s[1]), s[2]);
			}
			if(map_YES_DIRTY_AuthId_authName.size()==0){
				return false;
			}
			
			//<authName>!!!<authID>!!!<sourceChannel>
			TreeMap<Integer, String> map_eachLine_authName_AND_auth_id_txt_NO_dirtyAuthNames=
			readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
			readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
																			inFile_12_authName_AND_auth_id_txt_NO_dirtyAuthNames,
																			-1, 
																			-1, 
																			 "load bodytext",//debug_label,
																			 false // isPrintSOP
																			);
			System.out.println("LOADED map_eachLine_authName_AND_auth_id_txt_NO_dirtyAuthNames.SIZE:"
								+map_eachLine_authName_AND_auth_id_txt_NO_dirtyAuthNames.size()+" from:"+inFile_12_authName_AND_auth_id_txt_NO_dirtyAuthNames);;
			for(int seq:map_eachLine_authName_AND_auth_id_txt_NO_dirtyAuthNames.keySet()){
				String currLine=map_eachLine_authName_AND_auth_id_txt_NO_dirtyAuthNames.get(seq);
				String []s=currLine.split("!!!");
				map_authID_SourceChannel.put(Integer.valueOf(s[1]), s[2]);
			}
			// 
			TreeMap<Integer,String>  map_inFile9_eachLine=readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
					 									 .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
					 											 								inFile_9_auth_id_doc_id_queryTopicRelated__applying_multi_crossAuthIDs, 
																							 	 -1, //startline, 
																								 -1, //endline,
																								 " ", //debug_label
																								 false //isPrintSOP
																								 );
			// create map <authID,totalDocs>
			//authID!!!docID!!!query
			for(int seq_:map_inFile9_eachLine.keySet()){
				if(seq_==1) continue; //HEADER
				String line9=map_inFile9_eachLine.get(seq_);
				
				System.out.println("line9:"+line9);
				String []s=line9.split("!!!");
				int auth_id= Integer.valueOf(s[0]);
				int doc_id= Integer.valueOf(s[1]);
				
				//if DIRTY AUTHOR ID OF DIRTY AUTH NAME, SKIP
				if(map_YES_DIRTY_AuthId_authName.containsKey(auth_id)){
					continue;
				}
				
				if(!map_authID_AND_TotalDocs.containsKey(auth_id)){
					map_authID_AND_TotalDocs.put(auth_id, 1);
				}
				else{
					int t=map_authID_AND_TotalDocs.get(auth_id)+1;
					map_authID_AND_TotalDocs.put(auth_id, t);
				}
			}
			
			TreeMap<Integer,String>  map_inFile10_eachLine=readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
														 .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
																 								inFile_10_authName_AND_auth_id__MAPPEDmultiAuthID_SORTED_add_minAuthID, 
																							 	 -1, //startline, 
																								 -1, //endline,
																								 " ", //debug_label
																								 false //isPrintSOP
																								 );
			
			//approach 2
			if(flag_approach==2){
				//AuthName!!!AuthID!!!cross-MultiAuthIDmatched!!!distinctCount-cross-MultiAuthIDmatched!!!ALL_authIDs!!!min_AUTHID
				for(int seq_:map_inFile10_eachLine.keySet()){
					if(seq_==1) continue; //HEADER
					String line10=map_inFile10_eachLine.get(seq_);
					String [] s =line10.split("!!!");
					String authName=s[0];
					int authID=Integer.valueOf(s[1]);
					int min_authID=Integer.valueOf(s[5]);
					String ALL_authIDs=s[4];
					
					//if DIRTY AUTHOR ID OF DIRTY AUTH NAME, SKIP
					if(map_YES_DIRTY_AuthId_authName.containsKey(authID))
						continue;
					
					mapAuthId_authName.put(authID, authName);
					
					if(ALL_authIDs.indexOf(",")==-1)
						map_authID_minAuthId.put( Integer.valueOf(ALL_authIDs)  , min_authID);
					else{
						String [] s2=ALL_authIDs.split(",");
						int c=0;
						while(c < s2.length){
							map_authID_minAuthId.put( Integer.valueOf(s2[c])  , min_authID  );	
							c++;
						}
						
					}
					
				}
				//re-doing for map_authName_AND_freq as last iteration fills "map_authID_minAuthId"
				//AuthName!!!AuthID!!!cross-MultiAuthIDmatched!!!distinctCount-cross-MultiAuthIDmatched!!!ALL_authIDs!!!min_AUTHID
				for(int seq_:map_inFile10_eachLine.keySet()){
					if(seq_==1) continue; //HEADER
					String line10=map_inFile10_eachLine.get(seq_);
					String [] s =line10.split("!!!");
					String authName=s[0];
					int authID=Integer.valueOf(s[1]);
					
					if(!map_authID_minAuthId.containsKey( authID)){
						System.out.println("NOT EXISTS AUTHID...");
						continue;
					}
					
					System.out.println("s[1]:"+s[1]+" "+map_authID_minAuthId.containsKey(Integer.valueOf(s[1])));
					int min_authID= map_authID_minAuthId.get(authID);
					String ALL_authIDs=s[4];
					
					//if DIRTY AUTHOR ID OF DIRTY AUTH NAME, SKIP
					if(map_YES_DIRTY_AuthId_authName.containsKey(authID))
						continue;
				
//					System.out.println("map_authID_minAuthId:"+map_authID_minAuthId);
					map_authName_AND_freq.put(authName, map_authID_AND_TotalDocs.get(min_authID));
				}
			}
			
			////////////////
			
			FileWriter writerDebug=new FileWriter(new File(baseFolder+"debug_gt_2.txt"));
			System.out.println("inFile_output_Ground_Truth:"+inFile_output_Ground_Truth);
			BufferedReader reader_= new BufferedReader(new FileReader(new File(inFile_output_Ground_Truth)));
			
			BufferedReader reader_authName_freq=
					new BufferedReader(new FileReader(new File(inFile_outputFileName_inFile_3_keyword_AND_freq)));
			
			FileWriter writer_1=new FileWriter(new File(output_Ground_Truth_final_1));
			String output_Ground_Truth_final_1_TRAD=output_Ground_Truth_final_1+"_"+pattern_for_sourceChannel_TRAD+".txt";
			String output_Ground_Truth_final_1_POLI=output_Ground_Truth_final_1+"_"+pattern_for_sourceChannel_POLI+".txt";
			FileWriter writer_1_POLI=new FileWriter(new File(output_Ground_Truth_final_1_POLI));
			FileWriter writer_1_TRAD=new FileWriter(new File(output_Ground_Truth_final_1_TRAD));
			
			FileWriter writer_2=new FileWriter(new File(output_Ground_Truth_final_2));
			String output_Ground_Truth_final_2_TRAD=output_Ground_Truth_final_2+"_"+pattern_for_sourceChannel_TRAD+".txt";
			String output_Ground_Truth_final_2_POLI=output_Ground_Truth_final_2+"_"+pattern_for_sourceChannel_POLI+".txt";
			FileWriter writer_2_POLI=new FileWriter(new File(output_Ground_Truth_final_2_POLI));
			FileWriter writer_2_TRAD=new FileWriter(new File(output_Ground_Truth_final_2_TRAD));
			
			FileWriter writer_3=new FileWriter(new File(output_Ground_Truth_final_3)); //top N from output_Ground_Truth_final_1
			String output_Ground_Truth_final_3_POLI=output_Ground_Truth_final_3+"_"+pattern_for_sourceChannel_POLI+".txt";
			String output_Ground_Truth_final_3_TRAD=output_Ground_Truth_final_3+"_"+pattern_for_sourceChannel_TRAD+".txt";
			FileWriter writer_3_POLI=new FileWriter(new File(output_Ground_Truth_final_3_POLI));
			FileWriter writer_3_TRAD=new FileWriter(new File(output_Ground_Truth_final_3_TRAD));
			
			FileWriter writer_4=new FileWriter(new File(output_Ground_Truth_final_4)); //top N from output_Ground_Truth_final_2
			String output_Ground_Truth_final_4_POLI=output_Ground_Truth_final_4+"_"+pattern_for_sourceChannel_POLI+".txt";
			String output_Ground_Truth_final_4_TRAD=output_Ground_Truth_final_4+"_"+pattern_for_sourceChannel_TRAD+".txt";
			FileWriter writer_4_POLI=new FileWriter(new File(output_Ground_Truth_final_4_POLI));
			FileWriter writer_4_TRAD=new FileWriter(new File(output_Ground_Truth_final_4_TRAD));
			
			FileWriter writer_5=new FileWriter(new File(output_Ground_Truth_final_5)); //top N from output_Ground_Truth_final_2
			String output_Ground_Truth_final_5_POLI=output_Ground_Truth_final_5+"_"+pattern_for_sourceChannel_POLI+".txt";
			String output_Ground_Truth_final_5_TRAD=output_Ground_Truth_final_5+"_"+pattern_for_sourceChannel_TRAD+".txt";
			FileWriter writer_5_POLI=new FileWriter(new File(output_Ground_Truth_final_5_POLI));
			FileWriter writer_5_TRAD=new FileWriter(new File(output_Ground_Truth_final_5_TRAD));
			
			FileWriter writer_6=new FileWriter(new File(output_Ground_Truth_final_6)); //top N from output_Ground_Truth_final_2
			String output_Ground_Truth_final_6_POLI=output_Ground_Truth_final_6+"_"+pattern_for_sourceChannel_POLI+".txt";
			String output_Ground_Truth_final_6_TRAD=output_Ground_Truth_final_6+"_"+pattern_for_sourceChannel_TRAD+".txt";
			FileWriter writer_6_POLI=new FileWriter(new File(output_Ground_Truth_final_6_POLI));
			FileWriter writer_6_TRAD=new FileWriter(new File(output_Ground_Truth_final_6_TRAD));
			
			FileWriter writer_7=new FileWriter(new File(output_Ground_Truth_final_7));
			String output_Ground_Truth_final_7_POLI=output_Ground_Truth_final_7+"_"+pattern_for_sourceChannel_POLI+".txt";
			String output_Ground_Truth_final_7_TRAD=output_Ground_Truth_final_7+"_"+pattern_for_sourceChannel_TRAD+".txt";
			FileWriter writer_7_POLI=new FileWriter(new File(output_Ground_Truth_final_7_POLI));
			FileWriter writer_7_TRAD=new FileWriter(new File(output_Ground_Truth_final_7_TRAD));
			
			FileWriter writer_8=new FileWriter(new File(output_Ground_Truth_final_8));
			String output_Ground_Truth_final_8_POLI=output_Ground_Truth_final_8+"_"+pattern_for_sourceChannel_POLI+".txt";
			String output_Ground_Truth_final_8_TRAD=output_Ground_Truth_final_8+"_"+pattern_for_sourceChannel_TRAD+".txt";
			FileWriter writer_8_POLI=new FileWriter(new File(output_Ground_Truth_final_8_POLI));
			FileWriter writer_8_TRAD=new FileWriter(new File(output_Ground_Truth_final_8_TRAD));			
			
			FileWriter writer_authname_freq=new FileWriter(new File(output_Ground_Truth_auth_name_freq_final_5));
			
			TreeMap<String, String> mapUnique_q_topic_as_key=new TreeMap<String, String>();
			int lineNumber=0;
			//header
			writer_1.append("auth_ID!!!"+"topic"+"!!!"+"sum_SUM_tfIDF_and_Noun_count_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			writer_1_TRAD.append("auth_ID!!!"+"topic"+"!!!"+"sum_SUM_tfIDF_and_Noun_count_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			writer_1_POLI.append("auth_ID!!!"+"topic"+"!!!"+"sum_SUM_tfIDF_and_Noun_count_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			
			writer_2.append("auth_ID!!!"+"topic"+"!!!"+"sum_AVG_tfIDF_and_Noun_count_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			writer_2_TRAD.append("auth_ID!!!"+"topic"+"!!!"+"sum_AVG_tfIDF_and_Noun_count_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			writer_2_POLI.append("auth_ID!!!"+"topic"+"!!!"+"sum_AVG_tfIDF_and_Noun_count_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			
			writer_3.append("auth_ID!!!"+"topic"+"!!!"+"CUBIC_AVG_tfIDF_and_Noun_count_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			writer_3_TRAD.append("auth_ID!!!"+"topic"+"!!!"+"CUBIC_AVG_tfIDF_and_Noun_count_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			writer_3_POLI.append("auth_ID!!!"+"topic"+"!!!"+"CUBIC_AVG_tfIDF_and_Noun_count_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			
			writer_4.append("auth_ID!!!"+"topic"+"!!!"+"Npower_AVG_pers_orga_loc_num_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			writer_4_TRAD.append("auth_ID!!!"+"topic"+"!!!"+"Npower_AVG_pers_orga_loc_num_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			writer_4_POLI.append("auth_ID!!!"+"topic"+"!!!"+"Npower_AVG_pers_orga_loc_num_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			
			writer_5.append("auth_ID!!!"+"topic"+"!!!"+"AVG_orga_loc_num_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			writer_5_TRAD.append("auth_ID!!!"+"topic"+"!!!"+"AVG_orga_loc_num_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			writer_5_POLI.append("auth_ID!!!"+"topic"+"!!!"+"AVG_orga_loc_num_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			
			writer_6.append("auth_ID!!!"+"topic"+"!!!"+"AVG_loc_num_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			writer_6_TRAD.append("auth_ID!!!"+"topic"+"!!!"+"AVG_loc_num_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			writer_6_POLI.append("auth_ID!!!"+"topic"+"!!!"+"AVG_loc_num_and_currQtopic_count!!!authName!!!sourceChannel"+"\n");
			
			writer_7.append("auth_ID!!!"+"topic"+"!!!"+"docCount_for_qTopic!!!authName!!!sourceChannel"+"\n");
			writer_7_TRAD.append("auth_ID!!!"+"topic"+"!!!"+"docCount_for_qTopic!!!authName!!!sourceChannel"+"\n");
			writer_7_POLI.append("auth_ID!!!"+"topic"+"!!!"+"docCount_for_qTopic!!!authName!!!sourceChannel"+"\n");
			
			writer_8.append("auth_ID!!!"+"topic"+"!!!"+"docCount_for_qTopic!!!authName!!!sourceChannel"+"\n");
			writer_8_TRAD.append("auth_ID!!!"+"topic"+"!!!"+"docCount_for_qTopic!!!authName!!!sourceChannel"+"\n");
			writer_8_POLI.append("auth_ID!!!"+"topic"+"!!!"+"docCount_for_qTopic!!!authName!!!sourceChannel"+"\n");
			
			BufferedReader reader_authname_authID=
					new BufferedReader(new FileReader(new File(inFile_6_AuthorName_AND_Auth_ID)));
			
//			TreeMap<Integer, String> mapAuthId_sourceChannel=new TreeMap<Integer, String>();
			
			if(flag_approach==1){
				//readin author name and auth id
				while( (line=reader_authname_authID.readLine())!=null ){
					lineNumber++;
					if(lineNumber==1) continue; //header
					String [] s=line.split("!!!");
					
					if(isSOPprint)
						System.out.println(s[1] +"<->"+ s[0]);
					
					mapAuthId_authName.put(Integer.valueOf(s[1]), s[0]);
//					mapAuthId_sourceChannel.put(Integer.valueOf(s[1]), s[2]);
				}
			}
			
			lineNumber=0;
			//
			if(flag_approach==1){
				// each 
				while( (line=reader_authName_freq.readLine())!=null ){
					lineNumber++;
					if(lineNumber==1) continue; //header
					String [] s=line.split("!!!");
					String currName=p6_MAIN_Work.p6_normalize_auth_name(s[0]).replace("auth:from: ", "").replace("auth:from:","");
					
					
					//approach 1
					 //System.out.println(s[1] +"<->"+ s[0]);
					// this may have duplicates (same person but variant name because of PREFIX auth:from
					// NOT contains
					if(!map_authName_AND_freq.containsKey(currName)){
						map_authName_AND_freq.put(currName
												  , new Integer(s[1]));//AUTHOR NAME & FREQ
					}
					else{
						int ss=map_authName_AND_freq.get(s)+ new Integer(s[1]);
						map_authName_AND_freq.put(currName, ss);
					}
					//System.out.println("pair "+s[0]+"<->"+s[1]);
				}
			}
			
			lineNumber=0;
			//auth_ID(1)!!!Total_docs(2)!!!SUM_tf_idf(3)!!!AVG_tf_idf(4)!!!avgNOUNS(5)!!!tNOUNS(6)!!!Q_topic_Freq(7)!!!Author_Name(8)!!!curr_auth_cumm_SUM_person(9)!!!curr_auth_cumm_AVG_person(10)!!!curr_auth_cumm_SUM_organization(11)!!!curr_auth_cumm_AVG_organization(12)!!!curr_auth_cumm_SUM_location(13)!!!curr_auth_cumm_AVG_location(14)!!!curr_auth_cumm_SUM_numbers(15)!!!curr_auth_cumm_AVG_numbers(16)
			while( (line=reader_.readLine())!=null ){
				lineNumber++;
				String [] s=line.split("!!!");
				 
				String curr_Q_topic_Freq= s[6];
				
				if(lineNumber==1) continue; //SKIP HEADER
				
				//if DIRTY AUTHOR ID OF DIRTY AUTH NAME, SKIP
				if(map_YES_DIRTY_AuthId_authName.containsKey( Integer.valueOf(s[0]) ))
						continue;
				
				//skip first line for header
				if(lineNumber>1){ 
					map_each_q_topic_freq=
					convert_mapString_to_TreeMap(s[token_containing_mapString_each_Q_Topic_Freq-1]);
				}
				else{
					continue;
				}
				int currLine_total_documents=0;
				// get total documents from all queries. .example: auth_id:2670!!!score:19.0!!!auth name:webdesk@voanews.com (lisa schlein)!!!boko haram(9)!!!totalDocs:71!!!q_topic_count:{boko haram=19, climate change=6, crime=12, election=6, syria=28}!!!trad_all.txt
				for(String curr_q_topic2:map_each_q_topic_freq.keySet()){
					currLine_total_documents+=map_each_q_topic_freq.get(curr_q_topic2);
				}
				
				// each curr_q_topic
				for(String curr_q_topic:map_each_q_topic_freq.keySet()){
					
					// <authID,totalDocs> //read from another  file above
//					map_authID_AND_TotalDocs.put(Integer.valueOf(s[0]) , Integer.valueOf(s[1]) ); 
					
					double sum_1=new Double(s[2])+ //SUM_tf_idf
								 new Double(s[5])+ //tNOUNS
								 new Double(map_each_q_topic_freq.get(curr_q_topic)); //Q_topic_Freq
					
					double sum_2=new Double(s[3])+ //AVG_tf_idf
								 new Double(s[4])+ //avgNOUNS
								 new Double(map_each_q_topic_freq.get(curr_q_topic)); //Q_topic_Freq
					
					double sum_3=Math.cbrt(  new Double(s[3])* //AVG_tf_idf
											 new Double(s[4])* //avgNOUNS
											 new Double(map_each_q_topic_freq.get(curr_q_topic))); //Q_topic_Freq
					// Math.pow(x, 1/y)
					double sum_4=Math.sqrt(  new Double(s[9])* //curr_auth_cumm_AVG_person
							 				 new Double(s[11])* //curr_auth_cumm_AVG_organization
							 				 new Double(s[13])*	 //curr_auth_cumm_AVG_location
							 				 new Double(s[15]) ); //curr_auth_cumm_AVG_numbers
					
					double sum_5=Math.sqrt(  new Double(s[11])* //curr_auth_cumm_AVG_organization
							 				 new Double(s[13])*	 //curr_auth_cumm_AVG_location
							 				 new Double(s[15]) ); //curr_auth_cumm_AVG_numbers
					
					double sum_6=Math.sqrt(  new Double(s[13])*	 //curr_auth_cumm_AVG_location
							 				 new Double(s[15]) ); //curr_auth_cumm_AVG_numbers
					
					double sum_7=new Double(map_each_q_topic_freq.get(curr_q_topic)); 
					
					double sum_8=new Double(map_each_q_topic_freq.get(curr_q_topic)) + 
								   (new Double(map_each_q_topic_freq.get(curr_q_topic))/ (double) currLine_total_documents);
					
//					System.out.println("sum_8:"+currLine_total_documents+" "+sum_8);
//					writerDebug.append("sum_8:"+currLine_total_documents+" "+sum_8+ " "+map_each_q_topic_freq.get(curr_q_topic)+
//										" "+map_each_q_topic_freq+" "+curr_q_topic+
//										"\n");
//					writerDebug.flush();
					
					int authID= Integer.valueOf(s[0]);
					System.out.println("authID:"+authID+ " "+map_authID_minAuthId.containsKey(authID));
					int min_authID=-1;
					
					if(map_authID_minAuthId.containsKey(authID)  ){
						min_authID=map_authID_minAuthId.get(authID);
						
						//if DIRTY AUTHOR ID OF DIRTY AUTH NAME, SKIP
						if(map_YES_DIRTY_AuthId_authName.containsKey( min_authID ))
								continue;
						String newLine="";
						
						if(authID==min_authID){
							newLine=min_authID+"!!!"+curr_q_topic+"!!!"+sum_1+"!!!"+s[7]+"!!!"+curr_Q_topic_Freq+"!!!"+map_authID_SourceChannel.get(min_authID) 
													+"!!!"	+map_each_q_topic_freq+"\n";
							writer_1.append(newLine); 
							writer_1.flush();
							if(newLine.indexOf(pattern_for_sourceChannel_POLI)>=0){
								writer_1_POLI.append(newLine);
								writer_1_POLI.flush();
							}
							if(newLine.indexOf(pattern_for_sourceChannel_TRAD)>=0){
								writer_1_TRAD.append(newLine);
								writer_1_TRAD.flush();
							}
							
							newLine=min_authID+"!!!"+curr_q_topic+"!!!"+sum_2+"!!!"+s[7]+"!!!"+curr_Q_topic_Freq+"!!!"+map_authID_SourceChannel.get(min_authID) 
														+"!!!"	+map_each_q_topic_freq+"\n";
							writer_2.append(newLine);
							writer_2.flush();
							if(newLine.indexOf(pattern_for_sourceChannel_POLI)>=0){
								writer_2_POLI.append(newLine);
								writer_2_POLI.flush();
							}
							if(newLine.indexOf(pattern_for_sourceChannel_TRAD)>=0){
								writer_2_TRAD.append(newLine);
								writer_2_TRAD.flush();
							}
							
							newLine=min_authID+"!!!"+curr_q_topic+"!!!"+sum_3+"!!!"+s[7]+"!!!"+curr_Q_topic_Freq+"!!!"+map_authID_SourceChannel.get(min_authID)
												+"!!!"	+map_each_q_topic_freq+"\n";
							writer_3.append(newLine);
							writer_3.flush();
							if(newLine.indexOf(pattern_for_sourceChannel_POLI)>=0){
								writer_3_POLI.append(newLine);
								writer_3_POLI.flush();
							}
							if(newLine.indexOf(pattern_for_sourceChannel_TRAD)>=0){
								writer_3_TRAD.append(newLine);
								writer_3_TRAD.flush();
							}
		
							newLine=min_authID+"!!!"+curr_q_topic+"!!!"+sum_4+"!!!"+s[7]+"!!!"+curr_Q_topic_Freq+"!!!"+map_authID_SourceChannel.get(min_authID) 
												+"!!!"	+map_each_q_topic_freq+"\n";
							writer_4.append(newLine);
							writer_4.flush();
							if(newLine.indexOf(pattern_for_sourceChannel_POLI)>=0){
								writer_4_POLI.append(newLine);
								writer_4_POLI.flush();
							}
							if(newLine.indexOf(pattern_for_sourceChannel_TRAD)>=0){
								writer_4_TRAD.append(newLine);
								writer_4_TRAD.flush();
							}
							
							newLine=min_authID+"!!!"+curr_q_topic+"!!!"+sum_5+"!!!"+s[7]+"!!!"+curr_Q_topic_Freq+"!!!"+map_authID_SourceChannel.get(min_authID)
												+"!!!"	+map_each_q_topic_freq+"\n";
							writer_5.append(newLine);
							writer_5.flush();
							if(newLine.indexOf(pattern_for_sourceChannel_POLI)>=0){
								writer_5_POLI.append(newLine);
								writer_5_POLI.flush();
							}
							if(newLine.indexOf(pattern_for_sourceChannel_TRAD)>=0){
								writer_5_TRAD.append(newLine);
								writer_5_TRAD.flush();
							}
							
							newLine=min_authID+"!!!"+curr_q_topic+"!!!"+sum_6+"!!!"+s[7]+"!!!"+curr_Q_topic_Freq+"!!!"+map_authID_SourceChannel.get(min_authID) + 
									      	"!!!"	+map_each_q_topic_freq+"\n";
							writer_6.append(newLine);
							writer_6.flush();
							
							if(newLine.indexOf(pattern_for_sourceChannel_POLI)>=0){
								writer_6_POLI.append(newLine);
								writer_6_POLI.flush();
							}
							if(newLine.indexOf(pattern_for_sourceChannel_TRAD)>=0){
								writer_6_TRAD.append(newLine);
								writer_6_TRAD.flush();
							}
							// 7
							newLine=min_authID+"!!!"+curr_q_topic+"!!!"+sum_7+"!!!"+s[7]+"!!!"+curr_Q_topic_Freq+"!!!"+map_authID_SourceChannel.get(min_authID)
												+"!!!"	+map_each_q_topic_freq+"\n";
							writer_7.append(newLine);
							writer_7.flush();
							
							if(newLine.indexOf(pattern_for_sourceChannel_POLI)>=0){
								writer_7_POLI.append(newLine);
								writer_7_POLI.flush();
							}
							if(newLine.indexOf(pattern_for_sourceChannel_TRAD)>=0){
								writer_7_TRAD.append(newLine);
								writer_7_TRAD.flush();
							}
							// 8
							newLine=min_authID+"!!!"+curr_q_topic+"!!!"+sum_8+"!!!"+s[7]+"!!!"+curr_Q_topic_Freq+"!!!"+map_authID_SourceChannel.get(min_authID)
										        +"!!!"	+map_each_q_topic_freq+"\n";
							writer_8.append(newLine);
							writer_8.flush();
							
							if(newLine.indexOf(pattern_for_sourceChannel_POLI)>=0){
								writer_8_POLI.append(newLine);
								writer_8_POLI.flush();
							}
							if(newLine.indexOf(pattern_for_sourceChannel_TRAD)>=0){
								writer_8_TRAD.append(newLine);
								writer_8_TRAD.flush();
							}
							
						}
						else{ //DEBUG
							writerDebug.append("2.authID!=min_authID"+min_authID+"!!!"+curr_q_topic+"!!!"+sum_7+"!!!"+s[7]+"!!!"+curr_Q_topic_Freq+"\n");
							writerDebug.flush();
						}
					}
					
					if(!mapUnique_q_topic_as_key.containsKey(curr_q_topic)){
						mapUnique_q_topic_as_key.put(curr_q_topic, "");
					}
					
				} //END each curr_q_topic
			}
			
			
			//***************************output_Ground_Truth_final_1********************************
			String [] arr_inputFile=new String[2];
			arr_inputFile[0]=output_Ground_Truth_final_1_TRAD;
			arr_inputFile[1]=output_Ground_Truth_final_1_POLI;
			int c=0;
			//
			while(c<arr_inputFile.length){
				//OVERWRITE
				output_Ground_Truth_final_1=arr_inputFile[c];
			
			FileWriter write_10=new FileWriter(new File(output_Ground_Truth_final_1+"_Top_N.txt"));
			TreeMap<Integer,Integer> map_authID_alreadyWrote_as_KEY=new TreeMap<Integer, Integer>();
			// EACH CURR_Q_TOPIC
			for(String curr_q_topic:mapUnique_q_topic_as_key.keySet()){
				map_authID_alreadyWrote_as_KEY=new TreeMap<Integer, Integer>();
				reader_=new BufferedReader(new FileReader(new File(output_Ground_Truth_final_1)));
				lineNumber=0;
				TreeMap<Integer, Integer> map_lineNo_authid= new TreeMap<Integer, Integer>();
				TreeMap<Integer, String> map_lineNo_q_topic= new TreeMap<Integer, String>();
				TreeMap<Integer, Double> map_lineNo_score= new TreeMap<Integer, Double>();
				TreeMap<Integer, String> map_lineNo_Q_topic_Freq= new TreeMap<Integer, String>();
				TreeMap<Integer, String> map_lineNo_sourceChannel= new TreeMap<Integer, String>();
			
				// FOR each Q_topic, get Top N
				while( (line=reader_.readLine())!=null ){
					lineNumber++;
					if(line.indexOf(curr_q_topic)==-1) continue; //filter for only curr q topic
					String [] s=line.split("!!!");
					//
					if(lineNumber==1){ continue; }
					
					map_lineNo_authid.put(lineNumber, new Integer(s[0]));
					map_lineNo_q_topic.put(lineNumber, s[1]);
					map_lineNo_score.put(lineNumber, new Double(s[2]));
					map_lineNo_Q_topic_Freq.put(lineNumber, (s[4]));
					map_lineNo_sourceChannel.put(lineNumber, (s[5]));
				}
				TreeMap<Integer,Double> map=new TreeMap<Integer, Double>();
//				TreeMap<Integer,Integer> map= sort_given_treemap.sortByValue(map_lineNo_scoreINTEGER);
				Map<Integer,Double> map_orig= sort_given_treemap.sortByValue_ID_method4_ascending(
																							map_lineNo_score);
				//create map_reverse
		        ArrayList<Integer> keys = new ArrayList<Integer>(map_orig.keySet());
		        for(int i=keys.size()-1; i>=0;i--){
//		            System.out.println(map.get(keys.get(i)));
		            map.put(keys.get(i), map_orig.get(keys.get(i)));
		        }
				//System.out.println(map+"\n"+map_lineNo_authid);
				System.out.println("------------top N for Q topic :"+curr_q_topic);
				write_10.append("------------top N for Q topic :"+curr_q_topic+"\n");
				//top 5 read
				int c2=0;
//				for(int i:map.keySet() ){
//				int authID=map_lineNo_authid.get(i);
				for(int h=keys.size()-1; h>=0;h--){
					int i=keys.get(h);
					int authID=map_lineNo_authid.get(i);
					
					if(map_authID_alreadyWrote_as_KEY.containsKey(authID))
						continue;
					
//					System.out.println("auth_id:"+i+" score:"+map.get(i)
//									+" auth name:"+mapAuthId_authName.get(map_lineNo_authid.get(i)));
					
					write_10.append("auth_id:"+authID+"!!!score:"+map.get(i)
									+"!!!auth name:"+mapAuthId_authName.get(map_lineNo_authid.get(i))+
									"!!!"+curr_q_topic+"("+(c2+1)+")"+
									"!!!totalDocs:"+map_authID_AND_TotalDocs.get(authID)+
									"!!!q_topic_count:"+map_lineNo_Q_topic_Freq.get(i)+
									"!!!"+map_lineNo_sourceChannel.get(i)+
									"\n");
					write_10.flush();
					map_authID_alreadyWrote_as_KEY.put(authID, -1);
					
					if(c2>N_Top_authors) break;
					c2++;
				}
			} //END for(String curr_q_topic:mapUnique_q_topic_as_key.keySet()){
			
			
			c++;
		   }
			
			
			//***************************output_Ground_Truth_final_3********************************
			
			arr_inputFile[0]=output_Ground_Truth_final_3_TRAD;
			arr_inputFile[1]=output_Ground_Truth_final_3_POLI;
			c=0;
			//
			while(c<arr_inputFile.length){
			//OVERWRITE
			output_Ground_Truth_final_3=arr_inputFile[c];
				
			//START FORMAT 3
			FileWriter write_30=new FileWriter(new File(output_Ground_Truth_final_3+"_Top_N.txt"));
			TreeMap<Integer, Integer> map_authID_alreadyWrote_as_KEY=new TreeMap<Integer, Integer>();
			// EACH CURR_Q_TOPIC
			for(String curr_q_topic:mapUnique_q_topic_as_key.keySet()){
				map_authID_alreadyWrote_as_KEY=new TreeMap<Integer, Integer>();
				reader_=
						new BufferedReader(new FileReader(new File(output_Ground_Truth_final_3)));
				lineNumber=0;
				TreeMap<Integer, Integer> map_lineNo_authid= new TreeMap<Integer, Integer>();
				TreeMap<Integer, String> map_lineNo_q_topic= new TreeMap<Integer, String>();
				TreeMap<Integer, Double> map_lineNo_score= new TreeMap<Integer, Double>();
				TreeMap<Integer, String> map_lineNo_Q_topic_Freq= new TreeMap<Integer, String>();
				TreeMap<Integer, String> map_lineNo_sourceChannel= new TreeMap<Integer, String>();
			
				// FOR each Q_topic, get Top N
				while( (line=reader_.readLine())!=null ){
					lineNumber++;
					if(line.indexOf(curr_q_topic)==-1) continue; //filter for only curr q topic
					String [] s=line.split("!!!");
					//
					if(lineNumber==1){ continue; }
					System.out.println("line:"+line);
					map_lineNo_authid.put(lineNumber, new Integer(s[0]));
					map_lineNo_q_topic.put(lineNumber, s[1]);
					map_lineNo_score.put(lineNumber, new Double(s[2]));
					map_lineNo_Q_topic_Freq.put(lineNumber, (s[4]));
					map_lineNo_sourceChannel.put(lineNumber, (s[5]));
				}
				TreeMap<Integer,Double> map=new TreeMap<Integer, Double>();
//				TreeMap<Integer,Integer> map= sort_given_treemap.sortByValue(map_lineNo_scoreINTEGER);
				Map<Integer,Double> map_orig= sort_given_treemap.sortByValue_ID_method4_ascending(
																							map_lineNo_score);
				//create map_reverse
		        ArrayList<Integer> keys = new ArrayList<Integer>(map_orig.keySet());
		        for(int i=keys.size()-1; i>=0;i--){
//		            System.out.println(map.get(keys.get(i)));
		            map.put(keys.get(i), map_orig.get(keys.get(i)));
		        }
				//System.out.println(map+"\n"+map_lineNo_authid);
				System.out.println("------------top N for Q topic :"+curr_q_topic);
				write_30.append("------------top N for Q topic :"+curr_q_topic+"\n");
				//top 5 read
				int c2=0;
//				for(int i:map.keySet() ){
//				int authID=map_lineNo_authid.get(i);
				for(int h=keys.size()-1; h>=0;h--){
					int i=keys.get(h);
					int authID=map_lineNo_authid.get(i);
//					System.out.println("auth_id:"+i+" score:"+map.get(i)
//									+" auth name:"+mapAuthId_authName.get(map_lineNo_authid.get(i)));
					
					if(map_authID_alreadyWrote_as_KEY.containsKey(authID))
						continue;
					
					write_30.append("auth_id:"+authID+"!!!score:"+map.get(i)
									+"!!!auth name:"+mapAuthId_authName.get(map_lineNo_authid.get(i))
									+"!!!"+curr_q_topic+"("+(c2+1)+")"
									+"!!!totalDocs:"+map_authID_AND_TotalDocs.get(authID)
									+"!!!q_topic_count:"+map_lineNo_Q_topic_Freq.get(i)
									+"!!!"+map_lineNo_sourceChannel.get(i)
									+"\n");
					write_30.flush();
					map_authID_alreadyWrote_as_KEY.put(authID, -1);
					
					if(c2>N_Top_authors) break;
					c2++;
				}
			} //END for(String curr_q_topic:mapUnique_q_topic_as_key.keySet()){
			
			c++;
			}
			
			//END FORMAT 3
			

			//**************************************output_Ground_Truth_final_4***********************************************
			
			arr_inputFile[0]=output_Ground_Truth_final_4_TRAD;
			arr_inputFile[1]=output_Ground_Truth_final_4_POLI;
			  c=0;
			//
			while(c<arr_inputFile.length){
				//OVERWRITE
				output_Ground_Truth_final_4=arr_inputFile[c];
				
			FileWriter write_40=new FileWriter(new File(output_Ground_Truth_final_4+"_Top_N.txt"));
			TreeMap<Integer, Integer> map_authID_alreadyWrote_as_KEY=new TreeMap<Integer, Integer>();
			// EACH CURR_Q_TOPIC
			for(String curr_q_topic:mapUnique_q_topic_as_key.keySet()){
				map_authID_alreadyWrote_as_KEY=new TreeMap<Integer, Integer>();
				reader_=
						new BufferedReader(new FileReader(new File(output_Ground_Truth_final_4)));
				lineNumber=0;
				TreeMap<Integer, Integer> map_lineNo_authid= new TreeMap<Integer, Integer>();
				TreeMap<Integer, String> map_lineNo_q_topic= new TreeMap<Integer, String>();
				TreeMap<Integer, Double> map_lineNo_score= new TreeMap<Integer, Double>();
				TreeMap<Integer, String> map_lineNo_Q_topic_Freq= new TreeMap<Integer, String>();
				TreeMap<Integer, String> map_lineNo_sourceChannel= new TreeMap<Integer, String>();
				
				// FOR each Q_topic, get Top N
				while( (line=reader_.readLine())!=null ){
					lineNumber++;
					if(line.indexOf(curr_q_topic)==-1) continue; //filter for only curr q topic
					String [] s=line.split("!!!");
					//
					if(lineNumber==1){ continue; }
					
					map_lineNo_authid.put(lineNumber, new Integer(s[0]));
					map_lineNo_q_topic.put(lineNumber, s[1]);
					map_lineNo_score.put(lineNumber, new Double(s[2]));
					map_lineNo_Q_topic_Freq.put(lineNumber, (s[4]));
					map_lineNo_sourceChannel.put(lineNumber, (s[5]));
				}
				TreeMap<Integer,Double> map=new TreeMap<Integer, Double>();
//				TreeMap<Integer,Integer> map= sort_given_treemap.sortByValue(map_lineNo_scoreINTEGER);
				Map<Integer,Double> map_orig= sort_given_treemap.sortByValue_ID_method4_ascending(
																							map_lineNo_score);
				//create map_reverse
		        ArrayList<Integer> keys = new ArrayList<Integer>(map_orig.keySet());
		        for(int i=keys.size()-1; i>=0;i--){
//		            System.out.println(map.get(keys.get(i)));
		            map.put(keys.get(i), map_orig.get(keys.get(i)));
		        }
				//System.out.println(map+"\n"+map_lineNo_authid);
				System.out.println("------------top N for Q topic :"+curr_q_topic);
				write_40.append("------------top N for Q topic :"+curr_q_topic+"\n");
				//top 5 read
				int c2=0;
//				for(int i:map.keySet() ){
//				int authID=map_lineNo_authid.get(i);
				for(int h=keys.size()-1; h>=0;h--){
					int i=keys.get(h);
					int authID=map_lineNo_authid.get(i);
//					System.out.println("auth_id:"+i+" score:"+map.get(i)
//									+" auth name:"+mapAuthId_authName.get(map_lineNo_authid.get(i)));
					
					if(map_authID_alreadyWrote_as_KEY.containsKey(authID))
						continue;
					
					write_40.append("auth_id:"+authID+"!!!score:"+map.get(i)
									+"!!!auth name:"+mapAuthId_authName.get(authID)
									+"!!!"+curr_q_topic+"("+(c2+1)+")"
									+"!!!totalDocs:"+map_authID_AND_TotalDocs.get(authID)
									+"!!!q_topic_count:"+map_lineNo_Q_topic_Freq.get(i)
									+"!!!"+map_lineNo_sourceChannel.get(i)
									+"\n");
					write_40.flush();
					map_authID_alreadyWrote_as_KEY.put(authID, -1);
					
					if(c2>N_Top_authors) break;
					c2++;
				}
			} //END for(String curr_q_topic:mapUnique_q_topic_as_key.keySet()){
			
			c++;
			}
			
			//**************************************output_Ground_Truth_final_5***********************************************
			
			arr_inputFile[0]=output_Ground_Truth_final_5_TRAD;
			arr_inputFile[1]=output_Ground_Truth_final_5_POLI;
			c=0;
			//
			while(c<arr_inputFile.length){
				//OVERWRITE
				output_Ground_Truth_final_5=arr_inputFile[c];
				
			FileWriter write_50=new FileWriter(new File(output_Ground_Truth_final_5+"_Top_N.txt"));
			TreeMap<Integer, Integer> map_authID_alreadyWrote_as_KEY=new TreeMap<Integer, Integer>();
			// EACH CURR_Q_TOPIC
			for(String curr_q_topic:mapUnique_q_topic_as_key.keySet()){
				map_authID_alreadyWrote_as_KEY=new TreeMap<Integer, Integer>();
				reader_=
						new BufferedReader(new FileReader(new File(output_Ground_Truth_final_5)));
				lineNumber=0;
				TreeMap<Integer, Integer> map_lineNo_authid= new TreeMap<Integer, Integer>();
				TreeMap<Integer, String> map_lineNo_q_topic= new TreeMap<Integer, String>();
				TreeMap<Integer, Double> map_lineNo_score= new TreeMap<Integer, Double>();
				TreeMap<Integer, String> map_lineNo_Q_topic_Freq= new TreeMap<Integer, String>();
				TreeMap<Integer, String> map_lineNo_sourceChannel= new TreeMap<Integer, String>();
			
				// FOR each Q_topic, get Top N
				while( (line=reader_.readLine())!=null ){
					lineNumber++;
					if(line.indexOf(curr_q_topic)==-1) continue; //filter for only curr q topic
					String [] s=line.split("!!!");
					//
					if(lineNumber==1){ continue; }
					
					map_lineNo_authid.put(lineNumber, new Integer(s[0]));
					map_lineNo_q_topic.put(lineNumber, s[1]);
					map_lineNo_score.put(lineNumber, new Double(s[2]));
					map_lineNo_Q_topic_Freq.put(lineNumber, (s[4]));
					map_lineNo_sourceChannel.put(lineNumber, (s[5]));
				}
				TreeMap<Integer,Double> map=new TreeMap<Integer, Double>();
//				TreeMap<Integer,Integer> map= sort_given_treemap.sortByValue(map_lineNo_scoreINTEGER);
				Map<Integer,Double> map_orig= sort_given_treemap.sortByValue_ID_method4_ascending(
																							map_lineNo_score);
				//create map_reverse
		        ArrayList<Integer> keys = new ArrayList<Integer>(map_orig.keySet());
		        for(int i=keys.size()-1; i>=0;i--){
//		            System.out.println(map.get(keys.get(i)));
		            map.put(keys.get(i), map_orig.get(keys.get(i)));
		        }
				//System.out.println(map+"\n"+map_lineNo_authid);
				System.out.println("------------top N for Q topic :"+curr_q_topic);
				write_50.append("------------top N for Q topic :"+curr_q_topic+"\n");
				//top 5 read
				int c2=0;
//				for(int i:map.keySet() ){
//				int authID=map_lineNo_authid.get(i);
				for(int h=keys.size()-1; h>=0;h--){
					int i=keys.get(h);
					int authID=map_lineNo_authid.get(i);
//					System.out.println("auth_id:"+i+" score:"+map.get(i)
//									+" auth name:"+mapAuthId_authName.get(map_lineNo_authid.get(i)));
					
					if(map_authID_alreadyWrote_as_KEY.containsKey(authID))
						continue;
					
					write_50.append("auth_id:"+authID+"!!!score:"+map.get(i)
									+"!!!auth name:"+mapAuthId_authName.get(authID)
									+"!!!"+curr_q_topic+"("+(c2+1)+")"
									+"!!!totalDocs:"+map_authID_AND_TotalDocs.get(authID)
									+"!!!q_topic_count:"+map_lineNo_Q_topic_Freq.get(i)
									+"!!!"+map_lineNo_sourceChannel.get(i)
									+"\n");
					write_50.flush();
					map_authID_alreadyWrote_as_KEY.put(authID, -1);
					
					if(c2>N_Top_authors) break;
					c2++;
				}
			} //END for(String curr_q_topic:mapUnique_q_topic_as_key.keySet()){
			 
			
			c++;
			}
			
			//**************************************output_Ground_Truth_final_6***********************************************
			
			arr_inputFile[0]=output_Ground_Truth_final_6_TRAD;
			arr_inputFile[1]=output_Ground_Truth_final_6_POLI;
			c=0;
			//
			while(c<arr_inputFile.length){
				//OVERWRITE
				output_Ground_Truth_final_6=arr_inputFile[c];
				
			FileWriter write_60=new FileWriter(new File(output_Ground_Truth_final_6+"_Top_N.txt"));
			TreeMap<Integer, Integer> map_authID_alreadyWrote_as_KEY=new TreeMap<Integer, Integer>();
			// EACH CURR_Q_TOPIC
			for(String curr_q_topic:mapUnique_q_topic_as_key.keySet()){
				map_authID_alreadyWrote_as_KEY=new TreeMap<Integer, Integer>();
				reader_=
						new BufferedReader(new FileReader(new File(output_Ground_Truth_final_6)));
				lineNumber=0;
				TreeMap<Integer, Integer> map_lineNo_authid= new TreeMap<Integer, Integer>();
				TreeMap<Integer, String> map_lineNo_q_topic= new TreeMap<Integer, String>();
				TreeMap<Integer, Double> map_lineNo_score= new TreeMap<Integer, Double>();
				TreeMap<Integer, String> map_lineNo_Q_topic_Freq= new TreeMap<Integer, String>();
				TreeMap<Integer, String> map_lineNo_sourceChannel= new TreeMap<Integer, String>();
			
				// FOR each Q_topic, get Top N
				while( (line=reader_.readLine())!=null ){
					lineNumber++;
					if(line.indexOf(curr_q_topic)==-1) continue; //filter for only curr q topic
					String [] s=line.split("!!!");
					//
					if(lineNumber==1){ continue; }
					
					map_lineNo_authid.put(lineNumber, new Integer(s[0]));
					map_lineNo_q_topic.put(lineNumber, s[1]);
					map_lineNo_score.put(lineNumber, new Double(s[2]));
					map_lineNo_Q_topic_Freq.put(lineNumber, (s[4]));
					map_lineNo_sourceChannel.put(lineNumber, (s[5]));
				}
				TreeMap<Integer,Double> map=new TreeMap<Integer, Double>();
//				TreeMap<Integer,Integer> map= sort_given_treemap.sortByValue(map_lineNo_scoreINTEGER);
				Map<Integer,Double> map_orig= sort_given_treemap.sortByValue_ID_method4_ascending(
																							map_lineNo_score);
				//create map_reverse
		        ArrayList<Integer> keys = new ArrayList<Integer>(map_orig.keySet());
		        for(int i=keys.size()-1; i>=0;i--){
//		            System.out.println(map.get(keys.get(i)));
		            map.put(keys.get(i), map_orig.get(keys.get(i)));
		        }
				//System.out.println(map+"\n"+map_lineNo_authid);
				System.out.println("------------top N for Q topic :"+curr_q_topic);
				write_60.append("------------top N for Q topic :"+curr_q_topic+"\n");
				//top 5 read
				int c2=0;
//				for(int i:map.keySet() ){
//				int authID=map_lineNo_authid.get(i);
				for(int h=keys.size()-1; h>=0;h--){
					int i=keys.get(h);
					int authID=map_lineNo_authid.get(i);
//					System.out.println("auth_id:"+i+" score:"+map.get(i)
//									+" auth name:"+mapAuthId_authName.get(map_lineNo_authid.get(i)));
					if(map_authID_alreadyWrote_as_KEY.containsKey(authID))
						continue;
					
					write_60.append("auth_id:"+authID+"!!!score:"+map.get(i)
									+"!!!auth name:"+mapAuthId_authName.get(authID)
									+"!!!"+curr_q_topic+"("+(c2+1)+")"
									+"!!!totalDocs:"+map_authID_AND_TotalDocs.get(authID)
									+"!!!q_topic_count:"+map_lineNo_Q_topic_Freq.get(i)
									+"!!!"+map_lineNo_sourceChannel.get(i)
									+"\n");
					write_60.flush();
					map_authID_alreadyWrote_as_KEY.put(authID, -1);
					
					if(c2>N_Top_authors) break;
					c2++;
				}
			} //END for(String curr_q_topic:mapUnique_q_topic_as_key.keySet()){
			
			c++;
			}
			
			//**************************************output_Ground_Truth_final_7***********************************************
			
			arr_inputFile[0]=output_Ground_Truth_final_7_TRAD;
			arr_inputFile[1]=output_Ground_Truth_final_7_POLI;
			c=0;
			//
			while(c<arr_inputFile.length){
				//OVERWRITE
				output_Ground_Truth_final_7=arr_inputFile[c];
				
			FileWriter write_70=new FileWriter(new File(output_Ground_Truth_final_7+"_Top_N.txt"));
			
			// EACH CURR_Q_TOPIC
			for(String curr_q_topic:mapUnique_q_topic_as_key.keySet()){
				TreeMap<Integer,Integer> map_authID_alreadyWrote_as_KEY=new TreeMap<Integer, Integer>();
				reader_=new BufferedReader(new FileReader(new File(output_Ground_Truth_final_7)));
				lineNumber=0;
				TreeMap<Integer, Integer> map_lineNo_authid= new TreeMap<Integer, Integer>();
				TreeMap<Integer, String> map_lineNo_q_topic= new TreeMap<Integer, String>();
				TreeMap<Integer, Double> map_lineNo_score= new TreeMap<Integer, Double>();
				TreeMap<Integer, Integer> map_lineNo_scoreINTEGER= new TreeMap<Integer,Integer>();
				TreeMap<Integer, String> map_lineNo_Q_topic_Freq= new TreeMap<Integer, String>();
				TreeMap<Integer, String> map_lineNo_sourceChannel= new TreeMap<Integer, String>();
			
				// FOR each Q_topic, get Top N
				while( (line=reader_.readLine())!=null ){
					lineNumber++;
					if(line.indexOf(curr_q_topic)==-1) continue; //filter for only curr q topic
					String [] s=line.split("!!!");
					//
					if(lineNumber==1){ continue; }
					 
					//ONLY SELECTIVE curr query
					if(s[1].equalsIgnoreCase(curr_q_topic) ){
						map_lineNo_authid.put(lineNumber, new Integer(s[0]));
						map_lineNo_q_topic.put(lineNumber, s[1]);
						map_lineNo_score.put(lineNumber, new Double(s[2]));
						map_lineNo_scoreINTEGER.put(lineNumber, new Integer(s[2].substring(0, s[2].indexOf(".")) ));
						map_lineNo_Q_topic_Freq.put(lineNumber, (s[4]));
						map_lineNo_sourceChannel.put(lineNumber, (s[5]));
					}
				}
				  
				TreeMap<Integer,Double> map=new TreeMap<Integer, Double>();
//				TreeMap<Integer,Integer> map= sort_given_treemap.sortByValue(map_lineNo_scoreINTEGER);
				Map<Integer,Double> map_orig= sort_given_treemap.sortByValue_ID_method4_ascending(
																							map_lineNo_score);
				//create map_reverse
		        ArrayList<Integer> keys = new ArrayList<Integer>(map_orig.keySet());
		        for(int i=keys.size()-1; i>=0;i--){
//		            System.out.println(map.get(keys.get(i)));
		            map.put(keys.get(i), map_orig.get(keys.get(i)));
		        }
				
				writerDebug.append("\n Total lines:"+lineNumber +" curr_q_topic:"+curr_q_topic+ 
								" from file:"+output_Ground_Truth_final_7+"\n map(sorted map_lineNo_score):"+map
												+"\n map(not sorted):"+map_orig
												+"\n map.size:"+map.size()
												+" map_lineNo_score.size:"+map_lineNo_score.size());
				writerDebug.append("\n-------------------");
				writerDebug.flush();
				
				//System.out.println(map+"\n"+map_lineNo_authid);
				System.out.println("------------top N for Q topic :"+curr_q_topic);
				write_70.append("------------top N for Q topic :"+curr_q_topic+"\n");
				//top 5 read
				int c2=0;
//				keys = new ArrayList<Integer>(map_orig.keySet());
//				for(int i:map.keySet() ){
//				int authID=map_lineNo_authid.get(i);
				for(int h=keys.size()-1; h>=0;h--){
					int i=keys.get(h);
					int authID=	map_lineNo_authid.get(i);
//					System.out.println("auth_id:"+i+" score:"+map.get(i)
//									+" auth name:"+mapAuthId_authName.get(map_lineNo_authid.get(i)));
					
		            if(curr_q_topic.indexOf("india")>=0) {//debug
		            	writerDebug.append(" before..."+authID+"\n");
		            	writerDebug.flush();
		            }
					
					if(map_authID_alreadyWrote_as_KEY.containsKey(authID)){
						writerDebug.append("SKIPPING authID:"+authID+"\n");
						writerDebug.flush();
						continue;
					}
					
					if(curr_q_topic.indexOf("india")>=0) {//debug
		            	writerDebug.append(" after..."+authID+"\n");
		            	writerDebug.flush();
		            }
					
					
					write_70.append("auth_id:"+authID
									+"!!!score:"+map.get(i)
									+"!!!auth name:"+mapAuthId_authName.get(authID)
									+"!!!"+curr_q_topic+"("+(c2+1)+")"
									+"!!!totalDocs:"+map_authID_AND_TotalDocs.get(authID)
									+"!!!q_topic_count:"+map_lineNo_Q_topic_Freq.get(i)
									+"!!!"+map_lineNo_sourceChannel.get(i)
//									+"!!!"+map.size()+"!!!"+N_Top_authors+"!!!"+c2 <--debug
									+"\n");
					write_70.flush();
					map_authID_alreadyWrote_as_KEY.put(authID, -1);
					
					if(c2>N_Top_authors){

						writerDebug.append("\n 7.map_lineNo_score:"+map_lineNo_score);
						writerDebug.flush();
						break;
						 
					}
					c2++;
				}
			} //END for(String curr_q_topic:mapUnique_q_topic_as_key.keySet()){

			
			c++;
			}
			
			//**************************************output_Ground_Truth_final_8***********************************************
			
			arr_inputFile[0]=output_Ground_Truth_final_8_TRAD;
			arr_inputFile[1]=output_Ground_Truth_final_8_POLI;
			c=0;
			//
			while(c<arr_inputFile.length){
				//OVERWRITE
				output_Ground_Truth_final_8=arr_inputFile[c];
				
			FileWriter write_80=new FileWriter(new File(output_Ground_Truth_final_8+"_Top_N.txt"));
			
			writerDebug.append("\n output_Ground_Truth_final_8:"+output_Ground_Truth_final_8+"_Top_N.txt");
			writerDebug.flush();
			
			// EACH CURR_Q_TOPIC
			for(String curr_q_topic:mapUnique_q_topic_as_key.keySet()){
				TreeMap<Integer,Integer> map_authID_alreadyWrote_as_KEY=new TreeMap<Integer, Integer>();
				reader_=new BufferedReader(new FileReader(new File(output_Ground_Truth_final_8)));
				lineNumber=0;
				TreeMap<Integer, Integer> map_lineNo_authid= new TreeMap<Integer, Integer>();
				TreeMap<Integer, String> map_lineNo_q_topic= new TreeMap<Integer, String>();
				TreeMap<Integer, Double> map_lineNo_score= new TreeMap<Integer, Double>();
				TreeMap<Integer, Integer> map_lineNo_scoreINTEGER= new TreeMap<Integer,Integer>();
				TreeMap<Integer, String> map_lineNo_Q_topic_Freq= new TreeMap<Integer, String>();
				TreeMap<Integer, String> map_lineNo_sourceChannel= new TreeMap<Integer, String>();
			
				// FOR each Q_topic, get Top N
				while( (line=reader_.readLine())!=null ){
					lineNumber++;
					if(line.indexOf(curr_q_topic)==-1) continue; //filter for only curr q topic
					String [] s=line.split("!!!");
					//
					if(lineNumber==1){ continue; }
					 
					//ONLY SELECTIVE curr query
					if(s[1].equalsIgnoreCase(curr_q_topic) ){
						map_lineNo_authid.put(lineNumber, new Integer(s[0]));
						map_lineNo_q_topic.put(lineNumber, s[1]);
						map_lineNo_score.put(lineNumber, new Double(s[2]));
//						if(s[2].indexOf(".")==-1){
//							if(s[2].toLowerCase().indexOf("infinity")>=0){
//								map_lineNo_scoreINTEGER.put(lineNumber, 0 );
//							}
//							else{
//								map_lineNo_scoreINTEGER.put(lineNumber, new Integer(s[2]) );
//							}
//						}
//						else
							map_lineNo_scoreINTEGER.put(lineNumber, new Integer(s[2].substring(0, s[2].indexOf(".")) ));
						map_lineNo_Q_topic_Freq.put(lineNumber, (s[4]));
						map_lineNo_sourceChannel.put(lineNumber, (s[5]));
					}
				}
				  
				TreeMap<Integer,Double> map=new TreeMap<Integer, Double>();
//				TreeMap<Integer,Integer> map= sort_given_treemap.sortByValue(map_lineNo_scoreINTEGER);
				Map<Integer,Double> map_orig= sort_given_treemap.sortByValue_ID_method4_ascending(
																							map_lineNo_score);
				//create map_reverse
		        ArrayList<Integer> keys = new ArrayList<Integer>(map_orig.keySet());
		        for(int i=keys.size()-1; i>=0;i--){
//		            System.out.println(map.get(keys.get(i)));
		            map.put(keys.get(i), map_orig.get(keys.get(i)));
		        }
				
				writerDebug.append("\n Total lines:"+lineNumber +" curr_q_topic:"+curr_q_topic+ 
								" from file:"+output_Ground_Truth_final_8+"\n map(sorted map_lineNo_score):"+map
												+"\n map(not sorted):"+map_orig
												+"\n map.size:"+map.size()
												+" map_lineNo_score.size:"+map_lineNo_score.size());
				writerDebug.append("\n-------------------");
				writerDebug.flush();
				
				//System.out.println(map+"\n"+map_lineNo_authid);
				System.out.println("------------top N for Q topic :"+curr_q_topic);
				write_80.append("------------top N for Q topic :"+curr_q_topic+"\n");
				//top 5 read
				int c2=0;
//				keys = new ArrayList<Integer>(map_orig.keySet());
//				for(int i:map.keySet() ){
//				int authID=map_lineNo_authid.get(i);
				for(int h=keys.size()-1; h>=0;h--){
					int i=keys.get(h);
					int authID=	map_lineNo_authid.get(i);
//					System.out.println("auth_id:"+i+" score:"+map.get(i)
//									+" auth name:"+mapAuthId_authName.get(map_lineNo_authid.get(i)));
					
		            if(curr_q_topic.indexOf("india")>=0) {//debug
		            	writerDebug.append(" before..."+authID+"\n");
		            	writerDebug.flush();
		            }
					
					if(map_authID_alreadyWrote_as_KEY.containsKey(authID)){
						writerDebug.append("SKIPPING authID:"+authID+"\n");
						writerDebug.flush();
						continue;
					}
					
					if(curr_q_topic.indexOf("india")>=0) {//debug
		            	writerDebug.append(" after..."+authID+"\n");
		            	writerDebug.flush();
		            }
					
					
					write_80.append("auth_id:"+authID
									+"!!!score:"+map.get(i)
									+"!!!auth name:"+mapAuthId_authName.get(authID)
									+"!!!"+curr_q_topic+"("+(c2+1)+")"
									+"!!!totalDocs:"+map_authID_AND_TotalDocs.get(authID)
									+"!!!q_topic_count:"+map_lineNo_Q_topic_Freq.get(i)
									+"!!!"+map_lineNo_sourceChannel.get(i)
//									+"!!!"+map.size()+"!!!"+N_Top_authors+"!!!"+c2 <--debug
									+"\n");
					write_80.flush();
					map_authID_alreadyWrote_as_KEY.put(authID, -1);
					
					if(c2>N_Top_authors){

						writerDebug.append("\n 8.map_lineNo_score:"+map_lineNo_score);
						writerDebug.flush();
						break;
						 
					}
					write_80.flush();
					c2++;
				}
			} //END for(String curr_q_topic:mapUnique_q_topic_as_key.keySet()){

			
			c++;
			}
			
			//*********************************output_Ground_Truth_final_2**************************
			
			TreeMap<String, Integer> map_curr_q_topic_Authors_only_AND_freq_of_dis_docs=new TreeMap<String, Integer>();
			TreeMap<String, Integer> map_curr_q_topic_authName_Freq= new TreeMap<String, Integer>();
			
			arr_inputFile[0]=output_Ground_Truth_final_2_TRAD;
			arr_inputFile[1]=output_Ground_Truth_final_2_POLI;
			int c20=0;
			//
			while(c20<arr_inputFile.length){
				//OVERWRITE
				output_Ground_Truth_final_2=arr_inputFile[c20];
				
				
			TreeMap<Integer, Integer>  map_authID_alreadyWrote_as_KEY=new TreeMap<Integer, Integer>();
			FileWriter write_20=new FileWriter(new File(output_Ground_Truth_final_2+"_Top_N.txt"));
			// EACH CURR_Q_TOPIC
			for(String curr_q_topic:mapUnique_q_topic_as_key.keySet()){
			
				reader_=
						new BufferedReader(new FileReader(new File(output_Ground_Truth_final_2)));
				lineNumber=0;
				TreeMap<Integer, Integer> map_lineNo_authid= new TreeMap<Integer, Integer>();
				TreeMap<Integer, String> map_lineNo_q_topic= new TreeMap<Integer, String>();
				TreeMap<Integer, Double> map_lineNo_score= new TreeMap<Integer, Double>();
				TreeMap<Integer, String> map_lineNo_Q_topic_Freq= new TreeMap<Integer, String>();
				map_curr_q_topic_authName_Freq= new TreeMap<String, Integer>();
				TreeMap<Integer, String> map_lineNo_sourceChannel= new TreeMap<Integer, String>();
			
				// Reading all file SCORE, FOR each Q_topic, get Top N
				while( (line=reader_.readLine())!=null ){
					lineNumber++;
					if(line.indexOf(curr_q_topic)==-1) continue; //filter for only curr q topic
					String [] s=line.split("!!!");
					//
					if(lineNumber==1){ continue; }
					
					map_lineNo_authid.put(lineNumber, new Integer(s[0]));
					map_lineNo_q_topic.put(lineNumber, s[1]);
					map_lineNo_score.put(lineNumber, new Double(s[2]));
					map_lineNo_Q_topic_Freq.put(lineNumber, (s[4]));
					map_lineNo_sourceChannel.put(lineNumber, (s[5]));
				}
				TreeMap<Integer,Double> map=new TreeMap<Integer, Double>();
//				TreeMap<Integer,Integer> map= sort_given_treemap.sortByValue(map_lineNo_scoreINTEGER);
				Map<Integer,Double> map_orig= sort_given_treemap.sortByValue_ID_method4_ascending(
																							map_lineNo_score);
				
				
				//create map_reverse
		        ArrayList<Integer> keys = new ArrayList<Integer>(map_orig.keySet());
		        for(int i=keys.size()-1; i>=0;i--){
//		            System.out.println(map.get(keys.get(i)));
		            map.put(keys.get(i), map_orig.get(keys.get(i)));
		        }
				//System.out.println(map+"\n"+map_lineNo_authid);
				System.out.println("------------top N for Q topic :"+curr_q_topic);
				write_20.append("------------top N for Q topic :"+curr_q_topic+"\n");
				//top 5 read
				int c2=0;
//				for(int i:map.keySet() ){
//				int authID=map_lineNo_authid.get(i);
				for(int h=keys.size()-1; h>=0;h--){
					int i=keys.get(h);
					int authID=map_lineNo_authid.get(i);
					
					if(map_authID_alreadyWrote_as_KEY.containsKey(authID))
						continue;
					
					String curr_auth_name=mapAuthId_authName.get(authID);
					if(curr_auth_name==null){
						writerDebug.append("auth name not found with auth id:"+map_lineNo_authid.get(i));
						writerDebug.flush();
						continue;
					}
					
					curr_auth_name=p6_MAIN_Work.p6_normalize_auth_name(curr_auth_name).replace("auth:from: ", "");
					
					if(c2>N_Top_authors){
						if(map_authName_AND_freq.containsKey(curr_auth_name)){
							
							if(map_authName_AND_freq.get(curr_auth_name)!=null)
								map_curr_q_topic_authName_Freq.put(
										curr_auth_name, 
										map_authName_AND_freq.get(curr_auth_name));
						}
						else{
							writerDebug.append("\nNOT FOUND authname.1:"+curr_auth_name
														+" map_authName_AND_freq.size:"+map_authName_AND_freq.size()
														+" "+map_authName_AND_freq.containsKey(curr_auth_name));
							writerDebug.flush();
						}
						continue; // no more write to out file, but get author_names
					}
					//
					if(map_authName_AND_freq.containsKey(curr_auth_name)){
						if(map_authName_AND_freq.get(curr_auth_name)!=null)
							map_curr_q_topic_authName_Freq.put(
																curr_auth_name, 
																map_authName_AND_freq.get(curr_auth_name));
					}
					else{
						writerDebug.append("\nNOT FOUND authname.2:"+curr_auth_name
											+" map_authName_AND_freq.size:"+map_authName_AND_freq.size()
											+" "+map_authName_AND_freq.containsKey(curr_auth_name));
						writerDebug.flush();
					}
					
//					System.out.println("auth_id:"+i+" score:"+map.get(i)
//									+" auth name:"+curr_auth_name);
					
					write_20.append("auth_id:"+authID+"!!!score:"+map.get(i)
									+"!!!auth name:"+curr_auth_name
									+"!!!"+curr_q_topic+"("+(c2+1)+")"
									+"!!!totalDocs:"+map_authID_AND_TotalDocs.get(authID)
									+"!!!q_topic_count:"+map_lineNo_Q_topic_Freq.get(i)
									+"!!!"+map_lineNo_sourceChannel.get(i)
									+"\n");
					write_20.flush();
					map_authID_alreadyWrote_as_KEY.put(authID, -1);
					c2++;
				}
			
				writerDebug.append("\n curr_q_topic:"+curr_q_topic +" map_curr_q_topic_authName_Freq.size:"+map_curr_q_topic_authName_Freq.size()
				  +" map_curr_q_topic_authName_Freq:"+map_curr_q_topic_authName_Freq);
				writer_authname_freq.append("----------------curr_q_topic:"+curr_q_topic+"\n");
				writer_authname_freq.flush();
				
				//AT LAST RUN, WRITE TO AUTHOR FREQUENCY
//				if(c==arr_inputFile.length-1){
					//******************************auth frequency******sorted******for*****DEBUG**************************************
					//sort auth_name and frequency by freq
					TreeMap<String,Integer> map_authname_freq_sorted= new TreeMap<String, Integer>();
					
					if(map_curr_q_topic_authName_Freq !=null){
						if(map_curr_q_topic_authName_Freq.size()>0){
						
							System.out.println("map_curr_q_topic_authName_Freq:"+map_curr_q_topic_authName_Freq);
						map_authname_freq_sorted=(TreeMap<String, Integer>) sort_given_treemap.sortByValue_SI(map_curr_q_topic_authName_Freq);
//						System.out.println("--------------------auth freq for curr_q_topic:"+curr_q_topic+" map_authname_freq_sorted:"+map_authname_freq_sorted);
//						System.out.println("map_authName_AND_freq:"+map_authName_AND_freq);
						for(String i:map_authname_freq_sorted.keySet()){
							System.out.println(i+"!!!"+map_authname_freq_sorted.get(i));
							writer_authname_freq.append(i+"!!!"+map_authname_freq_sorted.get(i)+"!!!"+curr_q_topic+"\n");
							writer_authname_freq.flush();
						}
						}
					}
//				}
				
//				c++;
			 }
				
	
				c20++;
			} //END for(String curr_q_topic:mapUnique_q_topic_as_key.keySet()){
			
			writerDebug.append("\nmap_authName_AND_freq:"+map_authName_AND_freq);
			writerDebug.append("\nmap_authID_SourceChannel: from file->"+inFile_11_authName_AND_auth_id_txt_YES_dirtyAuthNames+" -->"+
									map_authID_SourceChannel);
			writerDebug.flush();
			
			for(String name:map_authName_AND_freq.keySet()){
				writerDebug.append("\n pair.n:"+name);
				writerDebug.flush();
			}
			
			System.out.println("map_authID_minAuthId:"+map_authID_minAuthId);

//			System.out.println("map_YES_DIRTY_AuthId_authName:"+map_YES_DIRTY_AuthId_authName);
		}
		catch(Exception e){
			e.printStackTrace();
		}
		return true;
	}
	// calc ground truth (NOT TESTED)
	private static boolean calc_ground_truth_1_2(String inFile_3_with_all_tokens,
			int token_no_contain_authors_in_inFile_3,
			int token_no_contain_bodyText_in_inFile_3,
			String CSV_containing_10_interested_topics,
			String outputFileName_inFile_3_keyword_AND_freq
			) {
			// TODO Auto-generated method stub
			TreeMap<String, Integer> mapUnique_Authors_Freq=new TreeMap<String, Integer>();
			try{
			
			TreeMap<String, String> mapUnique_Authors_Freq_EachLine=new TreeMap<String, String>();
			try{
					mapUnique_Authors_Freq_EachLine=
					readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
					readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile(
							outputFileName_inFile_3_keyword_AND_freq,
							-1, 
							-1, 
							"",//outFile  
							false, //is_Append_outFile, 
							false,//is_Write_To_OutputFile, 
							"debug_label", //, 
							-1, // token for  Primary key
							false // isSOPprint
							);
			}
			catch(Exception e){
					System.out.println("step 1...");
					e.printStackTrace();
			}
			
			//
			for(String curr_auth_freq:mapUnique_Authors_Freq_EachLine.keySet() ){
					String []s=curr_auth_freq.split("!!!");
					mapUnique_Authors_Freq.put(s[0],  Integer.valueOf(s[1]) );
			}
			
			TreeMap<Integer, String> map_eachLine_as_doc_all_tokens=new TreeMap<Integer, String>();
			try{
					String[] given_N_topics=CSV_containing_10_interested_topics.split(",");
					
					map_eachLine_as_doc_all_tokens=
					readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
					readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
								inFile_3_with_all_tokens,
								-1, 
								-1, 
								 "load bodytext",//debug_label,
								 true // isPrintSOP
								);
			}
			catch(Exception e){
					System.out.println("load failed map_eachLine_as_doc_all_tokens...");
					e.printStackTrace();
					return false;
			}
			
			TreeMap<Integer, String> map_eachLine_as_doc_AuthorName_AND_bodyText=new TreeMap<Integer, String>();
			//
			for(int lineNo:map_eachLine_as_doc_all_tokens.keySet()){
			
					String []s=map_eachLine_as_doc_all_tokens.get(lineNo).split("!!!");
					//if(lineNo<50000) { System.out.println("SKIPPING: lineNo:"+lineNo );continue;} //FAST DEBUG
					System.out.println("line22:"+map_eachLine_as_doc_all_tokens.get(lineNo));
					String curr_auth_name=p6_MAIN_Work.p6_normalize_auth_name(s[token_no_contain_authors_in_inFile_3-1]).replace("auth:from: ", "");
					String curr_bodyText=s[token_no_contain_bodyText_in_inFile_3-1];
					
					map_eachLine_as_doc_AuthorName_AND_bodyText.put(lineNo, curr_auth_name+" "+curr_bodyText );
			
			
			//System.out.println("concatenate->"+s[token_no_contain_authors_in_inFile_3-1]+" "
			//											+ s[token_no_contain_bodyText_in_inFile_3-1]);
			}//end for 
			}
			catch(Exception e){
				e.printStackTrace();
			}
			return true;
}

	// calc_ground_truth_compare_LDA_OR_hlda_plsi
	private static void calc_ground_truth_compare_LDA_OR_hlda_plsi( 
																	String baseFolder,
																	String inFile_1_top_N_groundTruth_file, 
																	String inFile_2_lda_R_hlda_R_plsi_Top_N,
																	String CSV_containing_10_interested_topics,
																	int    Top_N_to_match,
																	String outputFile, 
																	String outputFile_missing_AUTHID_while_MATCHING_from_Flag_3,
																	boolean is_Append_4_outputFile_missing_AUTHID_while_MATCHING_from_Flag_3
																	){
		BufferedReader reader_groundTruth=null;
		BufferedReader reader_lda_R_hlda_R_plsi_Top_N=null;
		String [] arr_CSV_containing_10_interested_topics=CSV_containing_10_interested_topics.split(",");
		FileWriter writer=null; FileWriter writerDebug=null;
		int count_matched=0;
		TreeMap <String, String > mapCountDistinctWritingQtopicASkey=new TreeMap<String, String>();
		TreeMap <String, String > map_already_wrote_line_MISSINGauthid=new TreeMap<String, String>();
		TreeMap <String, String > map_authID_queryCount_GT=new TreeMap<String, String>();
		TreeMap <String, String > map_UNIQUE_TopicANDauthName_for_GT_N_line=new TreeMap<String, String>();
		TreeMap <String, String > map_UNIQUE_TopicANDauthName_for_PREDICTED_N_line=new TreeMap<String, String>();
		TreeMap<Integer,Integer>  map_Unique_AuthID_GT_as_KEY = new TreeMap<Integer, Integer>();
		TreeMap<Integer,Integer>  map_Unique_AuthID_PRED_as_KEY = new TreeMap<Integer, Integer>();
		// TODO Auto-generated method stub
		try{
			FileWriter writer_missing_AUTHID=new FileWriter(new File(outputFile_missing_AUTHID_while_MATCHING_from_Flag_3)
															,is_Append_4_outputFile_missing_AUTHID_while_MATCHING_from_Flag_3);
			writer=new FileWriter(new File(outputFile));
			writerDebug=new FileWriter(new File(baseFolder+"debug.match.txt"));
			int c=0;
			writer.append("authID!!!score!!!topicID!!!authName!!!PRED_curr_q_topic!!!fileName!!!GT_curr_q_topic!!!PRED_curr_q_topic_RESET!!!mapQueryCnt\n");
			//each q_topic
			while(c<arr_CSV_containing_10_interested_topics.length){
				String curr_q_topic=arr_CSV_containing_10_interested_topics[c];
				reader_groundTruth=new BufferedReader(new FileReader(new File(inFile_1_top_N_groundTruth_file)));
				// debug (temp) dont forget to comment later
//				if(c>0) break; 
//				writer.append("------------For Q topic:"+curr_q_topic+"\n");
				writer.flush();
				
				TreeMap<Integer, String[]> map_lda_R_hlda_plsi=new TreeMap<Integer, String[]>();
				TreeMap<Integer, String> map_lda_R_hlda_plsi_FULLLINE=new TreeMap<Integer, String>();
				int lineNumber=0; String line="";
				TreeMap<Integer, String[]> map_groundTruth=new TreeMap<Integer, String []>();
				TreeMap<Integer, String> map_groundTruth_FULLLINE=new TreeMap<Integer, String>();
				//auth_id:2692!!!score:72.0!!!auth name:webdesk@voanews.com (moki edwin kindzeka)!!!boko haram(1)!!!totalDocs:83!!!q_topic_count:{boko haram=72, climate change=1, crime=2, election=5, syria=3}!!!trad_all.txt
				//each line of reader_groundTruth
				while( (line=reader_groundTruth.readLine())!=null ){
					lineNumber++;
					String authID="-1";
					//if(lineNumber==1) continue; //header 
					if(line.indexOf("!!!"+curr_q_topic)>=0&& line.indexOf("!!!")>=0){
						// map_UNIQUE_TopicANDauthName -- GROUND TRUTH
						authID= line.split("!!!")[0].replace("auth name:", "").replace("auth_id:", "");;
						String temp=curr_q_topic+"!!!"+authID;
						if(!map_UNIQUE_TopicANDauthName_for_GT_N_line.containsKey(temp)){
							map_UNIQUE_TopicANDauthName_for_GT_N_line.put(temp,"");// line);
						}
						map_groundTruth.put(lineNumber, line.split("!!!"));
						map_groundTruth_FULLLINE.put(lineNumber, line);
					} 
					if(!authID.equalsIgnoreCase("-1"))
						map_Unique_AuthID_GT_as_KEY.put(Integer.valueOf(authID), -1);
					
					//
					String []curr_s_GT=line.split("!!!");
					//  get <authID,queryCOUNT>
					String authID_GT=curr_s_GT[0].replace("auth_id:", "");
					if(!map_authID_queryCount_GT.containsKey(authID_GT) && curr_s_GT.length>=5){
						map_authID_queryCount_GT.put(authID_GT, curr_s_GT[5].replace("q_topic_count:", ""));
					}
					
					 
				} // END while( (line=reader_groundTruth.readLine())!=null ){
				writerDebug.append("\n filtered lines on (gTruth)="+curr_q_topic+" is "+map_groundTruth.size());
				writerDebug.flush();
				
//				if(curr_q_topic.indexOf("india")>=0){
//					for(Integer s:map_groundTruth.keySet()){
//						String[] st=map_groundTruth.get(s);
//						writerDebug.append("\n map_groundTruth(india):");
//						writerDebug.flush();
//						int cm=0; String str_="";
//						while(cm<st.length){
//							str_=str_+"!!!"+st[cm];
//							cm++;
//						}
//						 
//						writerDebug.flush();
//					}
//				}
				
				reader_lda_R_hlda_R_plsi_Top_N=new BufferedReader(new FileReader(new File(inFile_2_lda_R_hlda_R_plsi_Top_N)));
				lineNumber=0;
				//each line of reader_lda_R_hlda_R_plsi_Top_N
				while( (line=reader_lda_R_hlda_R_plsi_Top_N.readLine())!=null ){
					lineNumber++;
					if(lineNumber==1) continue; //header
					String authID="-1";
					//
					if(line.indexOf("!!!"+curr_q_topic)>=0 && line.indexOf("!!!")>=0 ){
						// map_UNIQUE_TopicANDauthName -- PREDICTED						
						authID= line.split("!!!")[0].replace("auth name:", "");
						map_Unique_AuthID_PRED_as_KEY.put(Integer.valueOf(authID), -1);
						String temp=curr_q_topic+"!!!"+authID;
						if(!map_UNIQUE_TopicANDauthName_for_PREDICTED_N_line.containsKey(temp)){
							map_UNIQUE_TopicANDauthName_for_PREDICTED_N_line.put(temp, ""); //line);
						}
						
						map_lda_R_hlda_plsi.put(lineNumber, line.split("!!!"));
						map_lda_R_hlda_plsi_FULLLINE.put(lineNumber, line);
					}

					 
				}
				writerDebug.append("\n filtered lines LDA on="+curr_q_topic+" is "+map_lda_R_hlda_plsi.size());
				writerDebug.append("\n map_Unique_AuthID_GT_as_KEY:"+map_Unique_AuthID_GT_as_KEY);
				writerDebug.flush();
				
				//iterate each line of GT, get authID,query_N_count
//				for(int lineNo2:map_groundTruth.keySet()){
//					//example:auth_id:63!!!score:60.0!!!auth name:alan rappeport!!!india(1)!!!totalDocs:251!!!q_topic_count:{boko haram=42, climate change=27, crime=32, election=39, india=60, syria=51}!!!trad_all.txt
//					String []curr_s_GT=map_groundTruth.get(lineNo2);
//					// 
//					String authID_GT=curr_s_GT[0].replace("auth_id:", "");	
//					map_authID_queryCount_GT.put(authID_GT, curr_s_GT[5].replace("q_topic_count:", ""));
//				}
				writerDebug.append("\n map_authID_queryCount_GT:"+map_authID_queryCount_GT
									+"\n map_authID_queryCount_GT.SIZE:"+map_authID_queryCount_GT.size()
									+"\n from  inFile_1_top_N_groundTruth_file:"+inFile_1_top_N_groundTruth_file+"\n");
				writerDebug.flush();
				
				// match on reader_lda_R_hlda_R_plsi_Top_N and reader_groundTruth <auth_id mismatch between them> <use names to match>
				int cnt2=0;
				count_matched=0;boolean cond_1=false; boolean cond_2=false; boolean cond_3=false;
				// PREDICTED
				for(int lineNo:map_lda_R_hlda_plsi.keySet()){
//					auth_id!!!Prob!!!topic_id!!!authName!!!z_q_topic_manual!!!sourceChannel
//					861!!!14.366884181336195!!!0!!!emily!!!election(1)!!!trad_all.txt
					String []curr_PREDICT=map_lda_R_hlda_plsi.get(lineNo);
					String authID_PREDICT_LDAorPLSI=curr_PREDICT[0];
					
					String curr_name_from_lda=	p6_MAIN_Work.p6_normalize_auth_name(curr_PREDICT[3])
															.replace("auth:from: ", "").replace("auth:from:", "");
					
					//iterate each line of GT, search this name in GT
					for(int lineNo2:map_groundTruth.keySet()){
						//
						if(count_matched==(Top_N_to_match))
							break;
						// 
						if(map_groundTruth_FULLLINE.get(lineNo2).indexOf("!!!"+curr_q_topic)==-1 &&
						   map_lda_R_hlda_plsi_FULLLINE.get(lineNo).indexOf("!!!"+curr_q_topic) ==-1){
							continue;
						}
						//example:auth_id:63!!!score:60.0!!!auth name:alan rappeport!!!india(1)!!!totalDocs:251!!!q_topic_count:{boko haram=42, climate change=27, crime=32, election=39, india=60, syria=51}!!!trad_all.txt
						String []curr_s_GT=map_groundTruth.get(lineNo2);
						// 
						String authID_GT=curr_s_GT[0].replace("auth_id:", "");
//						String curr_map_query_counts=curr_s_GT[5].replace("q_topic_count:","");
//						if(curr_s[4].indexOf(curr_q_topic+"(")==-1 || curr_s_GT[3].indexOf(curr_q_topic+"(")==-1 )
//							continue;
						
						//System.out.println("GT:"+ map_groundTruth_FULLLINE.get(lineNo2));
						String curr_name_from_GT=p6_MAIN_Work.p6_normalize_auth_name(curr_s_GT[2])//3rd column has name
													    	 .replace("auth:from: ", "").replace("auth:from:", "").replace("auth name:", "");
//						System.out.println("gtruth:"+map_groundTruth_FULLLINE.get(lineNo2)+" 3:"+curr_s_GT[3]);
//						System.out.println("lda/plsi"+map_lda_R_hlda_plsi_FULLLINE.get(lineNo)+" 4:"+curr_PREDICT[4]);
						String orig_curr_s_4=curr_PREDICT[4];
						String orig_curr_s_GT_3=curr_s_GT[3];
						
						if(curr_PREDICT[4].indexOf("(")>=0)
							curr_PREDICT[4]=curr_PREDICT[4].substring(0,curr_PREDICT[4].indexOf("("));
						if(curr_s_GT[3].indexOf("(")>=0)
							curr_s_GT[3]=curr_s_GT[3].substring(0,curr_s_GT[3].indexOf("("));
						
//						if(map_groundTruth_FULLLINE.get(lineNo2).indexOf("!!!india")>=0
//									//&& curr_name_from_lda.indexOf("emm ")>=0
//									&& curr_name_from_GT.indexOf("emm ")>=0
//									){
//							writerDebug.append("\n caught india!!: "+" 1:GT(name):"+curr_name_from_GT +" (lnNO:"+lineNo2+")"
//												+" 2:lda(name):"+curr_name_from_lda+" (lnNO:"+lineNo+")"
//												+" 3::"+curr_s_GT[3]+" 4:"+curr_s[4]
//												+" 5:"+curr_q_topic
//												+" 6:"+curr_s[4].equalsIgnoreCase(curr_s_GT[3])
//												+" 7:"+curr_s[4].equalsIgnoreCase(curr_q_topic));
//
//							writerDebug.flush();
//						}
						
						if(curr_q_topic.indexOf("india")>=0 && curr_name_from_GT.equalsIgnoreCase("emily")){
							writerDebug.append("\n caught india: "+curr_name_from_lda+" curr_PREDICT[4]:"+curr_PREDICT[4]
													+" curr_s_GT[3]:"+curr_s_GT[3]+" curr_name_from_lda:"+curr_name_from_lda
													+" "+(curr_name_from_GT.equalsIgnoreCase(curr_name_from_lda) )
													);
							writerDebug.flush();
						}
						String temp=curr_q_topic+"!!!"+authID_PREDICT_LDAorPLSI;
						cond_1=false;   cond_2=false;   cond_3=false;
						cond_1=  (authID_PREDICT_LDAorPLSI.equalsIgnoreCase(authID_GT)
//								&& map_UNIQUE_TopicANDauthName_for_GT_N_line.containsKey(curr_q_topic+"!!!"+curr_name_from_lda)//AUTH NAME from PREDICT exists in GT
								&& curr_PREDICT[4].equalsIgnoreCase(curr_s_GT[3])
								&& curr_PREDICT[4].equalsIgnoreCase(curr_q_topic)); //CASE 1: AuthID with same query topic in both side found
						
						cond_2= (map_Unique_AuthID_GT_as_KEY.containsKey( Integer.valueOf(authID_PREDICT_LDAorPLSI )));
								
						if( !map_UNIQUE_TopicANDauthName_for_GT_N_line.containsKey(curr_q_topic+"!!!"+authID_PREDICT_LDAorPLSI) )
							{cond_3=false;}
						else {cond_3=true;}
						
						String curr_map_query_counts="";
						
//						System.out.println("authID_GT:"+authID_GT+" authID_PREDICT_LDAorPLSI:"+authID_PREDICT_LDAorPLSI);
//						if((curr_name_from_GT.indexOf(curr_name_from_lda)>=0 )
//						if((curr_name_from_GT.equalsIgnoreCase(curr_name_from_lda) )
						if(     cond_1 ||
								((cond_2==true && cond_3==false)) //CASE 2: AuthID with same query topic in both side found
						  ){
//							System.out.println(
//									map_lda_R_hlda_plsi_FULLLINE.get(lineNo) + "!!!GT:"+curr_name_from_GT //ground truth append
//									);
//							System.out.println("-------MATCHED:"+authID_GT);
							count_matched++;
							
							String curr_ground_Truth="-1";
							String [] arr_ground_truth=map_groundTruth_FULLLINE.get(lineNo2).split("!!!");
//							String curr_map_query_counts=arr_ground_truth[5].replace("q_topic_count:","");
							
							if(!map_authID_queryCount_GT.containsKey(authID_GT) && cond_1==true)
								curr_map_query_counts="{}";
							else if(cond_1==true)
								curr_map_query_counts=map_authID_queryCount_GT.get(authID_GT);
							
							writerDebug.append("\n ----MATCHED 1:"+cond_1+"--2:"+cond_2+"--"+"3:"+cond_3+" 4:"+curr_q_topic
											  +" 5:"+curr_name_from_lda+" 6:"+curr_map_query_counts
											  +" 7:"+authID_GT+" cond_1:"+cond_1);
							writerDebug.flush();
							// 
							if(cond_1==true){
								curr_ground_Truth=arr_ground_truth[3];
							}
							else if(cond_2==true && cond_3==false){
								curr_ground_Truth=curr_q_topic+"(9999)"; //AuthID exists, but does not have any document for "curr query topic"
								curr_map_query_counts=map_authID_queryCount_GT.get(authID_PREDICT_LDAorPLSI);
							}
							 
							mapCountDistinctWritingQtopicASkey.put(curr_PREDICT[4], curr_s_GT[3]);
							//RECOUNT the RANKING order for AUTHID OF PREDICTED RANKING NOT AVILABLE IN GROUND TRUTH
							String [] s2=map_lda_R_hlda_plsi_FULLLINE.get(lineNo).split("!!!");
							
							String t=	map_lda_R_hlda_plsi_FULLLINE.get(lineNo) +"!!!"+curr_ground_Truth
																				 +"!!!"+s2[4].substring(0, s2[4].indexOf("("))+"("+count_matched+")"
																				 +"!!!"+curr_map_query_counts;
							
							if(t.toLowerCase().indexOf("parth")>=0){
								writerDebug.append("\n partha: 1:"+cond_1+" 2:"+cond_2+" "+"3:"+cond_3+" 4:"+curr_q_topic +" 5:"+curr_map_query_counts
													);
								writerDebug.flush();
							}
							t=t.replace("!!!!!!", "!!!");
							
							//RECOUNT the RANKING order for GROUND TRUTH 
							writer.append(t+"\n");//
							writer.flush();
							curr_map_query_counts="{}";
							break; 
						}	
						else if(
								!map_UNIQUE_TopicANDauthName_for_GT_N_line.containsKey(curr_q_topic+"!!!"+authID_GT) &&
								curr_PREDICT[4].equalsIgnoreCase(curr_s_GT[3]) && 
								curr_PREDICT[4].equalsIgnoreCase(curr_q_topic)
								){
							// missing auth id
							if(!map_already_wrote_line_MISSINGauthid.containsKey(map_lda_R_hlda_plsi_FULLLINE.get(lineNo))){
								writer_missing_AUTHID.append(map_lda_R_hlda_plsi_FULLLINE.get(lineNo)+"\n");
								writer_missing_AUTHID.flush();
								map_already_wrote_line_MISSINGauthid.put(map_lda_R_hlda_plsi_FULLLINE.get(lineNo), "");
							}
							 
							System.out.println("NOT MATCHED 1:"+map_groundTruth.size()+" 2:"+map_lda_R_hlda_plsi.size()
											+" curr_s_GT[3]:"+curr_s_GT[3]+" curr_PREDICT[4]:"+curr_PREDICT[4] +" curr_q_topic:"+curr_q_topic
											+" curr_name_from_GT:"+curr_name_from_GT+" curr_name_from_lda:"+curr_name_from_lda
											+" map_UNIQUE_TopicANDauthName_for_GT_N_line.SIZE:"+map_UNIQUE_TopicANDauthName_for_GT_N_line.size());
//							writer.append(map_lda_R_hlda_plsi_FULLLINE.get(lineNo) + "!!!GT:"+curr_s_GT[3]+"not matched"
//									+" 1:"+curr_name_from_GT+" 2:"+curr_name_from_lda
//									+"\n");
							writer.flush();
						}
						//comment and changed below
//						if(cnt2>(Top_N_to_match-1))
//							break;

					}
					  cond_1=false;   cond_2=false;   cond_3=false;
					cnt2++;

				} // END for(int lineNo:map_lda_R_hlda_plsi.keySet()){
				
				System.out.println("FILTERED OUT on lda map_lda_R_hlda_plsi.size:"+map_lda_R_hlda_plsi.size());
				writerDebug.append("\nFILTERED OUT on lda map_lda_R_hlda_plsi.size:"+map_lda_R_hlda_plsi.size()+" for curr_Q_topic:"+curr_q_topic);
				writerDebug.flush();
				
				map_groundTruth=new TreeMap<Integer, String[]>();
				map_groundTruth_FULLLINE=new TreeMap<Integer, String>();
				map_lda_R_hlda_plsi=new TreeMap<Integer, String[]>();
				map_lda_R_hlda_plsi_FULLLINE=new TreeMap<Integer, String>();
				c++;
				
				writerDebug.append("\n curr_q_topic:"+curr_q_topic+" count_matched:"+count_matched);
				writerDebug.flush();
				
			} // END while(c<arr_CSV_containing_10_interested_topics.length){
			System.out.println("***mapCountDistinctWritingQtopicASkey.size="+mapCountDistinctWritingQtopicASkey.size()
								+"\n"+mapCountDistinctWritingQtopicASkey);
			
			int cnt100=0;
			// 
			while(cnt100 < arr_CSV_containing_10_interested_topics.length){
				System.out.println("\ndebug csv array:"+cnt100+" topic:"+arr_CSV_containing_10_interested_topics[cnt100]);
				writerDebug.append("\ndebug csv array:"+cnt100+" topic:"+arr_CSV_containing_10_interested_topics[cnt100]);
				writerDebug.flush();
				cnt100++;
			}
			writerDebug.append("\narr_CSV_containing_10_interested_topics.len:"+arr_CSV_containing_10_interested_topics.length);
			writerDebug.append("\n map_UNIQUE_TopicANDauthName_for_PREDICTED_N_line:"+map_UNIQUE_TopicANDauthName_for_PREDICTED_N_line
								+"\n map_UNIQUE_TopicANDauthName_for_GT_N_line:"+map_UNIQUE_TopicANDauthName_for_GT_N_line
								);
			
			writerDebug.flush();
			System.out.println("given inFile_1_top_N_groundTruth_file:"+inFile_1_top_N_groundTruth_file);
			
			
			System.out.println("*inFile_1_top_N_groundTruth_file:"+inFile_1_top_N_groundTruth_file);
			System.out.println("*inFile_2_lda_R_hlda_R_plsi_Top_N:"+inFile_2_lda_R_hlda_R_plsi_Top_N);
			
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}
	// mean average precision
	private static TreeMap<String, Double> calc_mean_avg_precision_MAP(
												    String  baseFolder,
													String  inFile_output_matched,
													int     Top_N_max_for_calc_MAP,
													String  output_results,
													boolean is_Append_output_results,
													boolean is_Append_for_debugFile,
													boolean isSOPprint,
													int		Flag_choose_METHODtype_DCG, // {1= ,2,3,4 }
													boolean is_convert_using_bucket_for_rankingScore,
													boolean is_use_original_ranking,
													String  debugFile
													){
		BufferedReader reader=null;
		TreeMap<String, Double> mapOut=new TreeMap<String, Double>();
		String line="";int lineNumber=0; String token="";int lineNumber2=0;
		FileWriter writerDebug=null;
		FileWriter writer=null;
		// TODO Auto-generated method stub
		try{
			
			writerDebug=new FileWriter(new File(debugFile // "debug_MAP.txt"
												) ,is_Append_for_debugFile );
			writer=new FileWriter(new File(output_results), is_Append_output_results);
			
			System.out.println("inFile_output_matched:"+inFile_output_matched);
			writerDebug.append("\ninFile_output_matched:"+inFile_output_matched);
			writerDebug.flush();
			
			int MAX_NUMBER_AUTHORS=1000; 
			String curr_q_topic=""; String last_q_topic="";
			int reset_counter=0; double avg_prec_Num=0.0;
			TreeMap<String,TreeMap<Integer, Integer>> map_currQtopic_GT_rank_vs_pred_rank=new TreeMap<String,TreeMap<Integer, Integer>>();
			TreeMap<Integer, Integer> map_GT_rank_vs_pred_rank=new TreeMap<Integer, Integer>();
			TreeMap<Integer, Integer> map_pred_rank_vs_GT_rank=new TreeMap<Integer, Integer>();
			
			TreeMap<Integer, Integer> map_i_relI=new TreeMap<Integer, Integer>();
			
			TreeMap<String, Double> map_query_nDCG=new TreeMap<String, Double>();
			
			double precision=0.0;
			double avg_precision=0.0;
			double mean_precision=0.0;
			double mean_avg_precision=0.0;
			TreeMap<String, Double> map_qTopicID_Precision=new TreeMap<String, Double>();
			TreeMap<String, Double> map_qTopicID_Mean_Precision=new TreeMap<String, Double>();
			reader=new BufferedReader(new FileReader(new File(inFile_output_matched)));
			int positionCount=0;
			
			// https://en.wikipedia.org/wiki/Discounted_cumulative_gain
			double DCG = 0.; double iDCG = 0.;
			
			double DCG_n=0.;int i=0; int rel_i = 0; int rel_1_for_NUMERATOR=0;// for DCG
			TreeMap<String, String> map_UNIQUE_query_from_inFile_as_KEY=new TreeMap<String, String>(); 
			
			// read and get NO OF LINES.
			while( (line=reader.readLine())!=null ){
				lineNumber2++;
			}
			
			reader=new BufferedReader(new FileReader(new File(inFile_output_matched)));
			//each line read
			while( (line=reader.readLine())!=null ){
				lineNumber++;
				//
				if(lineNumber==1) continue; //header
 
				String []s=line.split("!!!");
				 //split "britain(1)" into token "britain" and "1"<--PICKING 8th token "PRED_curr_q_topic_RESET"
				token=s[7];
				String author_Name=s[3];
//				if(lineNumber<=5)
//					System.out.println("token:"+token+ " lineNumber:"+lineNumber+" --"+line);
				
				curr_q_topic=token.substring(0, token.indexOf("("));
				map_UNIQUE_query_from_inFile_as_KEY.put(curr_q_topic, "");
				
				int pred_rank=new Integer(token.substring(token.indexOf("("),token.length()).replace("(", "").replace(")", ""));
				int GT_rank=new Integer(s[6].substring(s[6].indexOf("("),s[6].length()).replace("(", "").replace(")", ""));
				// DCG 
				int GT_rank_4_DCG=Top_N_max_for_calc_MAP - GT_rank;
				int pred_rank_4_DCG=0;
				
				if(lineNumber<=1)
					writerDebug.append("\nlineNumber:"+lineNumber+" token:"+token+" GT rank:"+GT_rank+" line:"+line
										+" inFile_output_matched:"+inFile_output_matched.substring( inFile_output_matched.lastIndexOf("/") )); 
			 
				writerDebug.flush();
				map_currQtopic_GT_rank_vs_pred_rank.put(curr_q_topic, map_GT_rank_vs_pred_rank);
				positionCount=0;
				List<Entry<Integer, Integer>> map_i_relI_SORTED_descend_INT=null;
				//
				if(!curr_q_topic.equalsIgnoreCase(last_q_topic) && last_q_topic.length()>0
						|| lineNumber==lineNumber2
						){
					int total_GT_rank=0;
					//total GROUND TRUTH RANK
					for(int pred_rnk2:map_pred_rank_vs_GT_rank.keySet()){
						total_GT_rank+=map_pred_rank_vs_GT_rank.get(pred_rnk2);
					}
					reset_counter=0; 
					avg_prec_Num=0.;
					System.out.println("**** change in curr_q_topic last_q_topic:"+last_q_topic+" curr_q_topic:"+curr_q_topic);
					writerDebug.append("\n**** change in curr_q_topic last_q_topic:"+last_q_topic+" curr_q_topic:"+curr_q_topic);
					writerDebug.append("\nmap_GT_vs_pred_rank:"+map_GT_rank_vs_pred_rank);
					int hit_count=0; int hit_counter=0; int TEMP_MAX_NUMBER_AUTHOR=0;
					//SORT by only PRED_RNK , iterate and calc map_numerator
					for(int pred_rnk:map_pred_rank_vs_GT_rank.keySet()){
						positionCount++;
						//  only if the PRED RANK is a HOT
//						if(map_pred_rank_vs_GT_rank.get(pred_rnk)<=Top_N_max_for_calc_MAP 
//						   && pred_rnk<=Top_N_max_for_calc_MAP){
//							i=positionCount;
//							//
//							if(is_convert_using_bucket_for_rankingScore==false)
//								pred_rank_4_DCG=Top_N_max_for_calc_MAP - pred_rank;
//							else if(is_convert_using_bucket_for_rankingScore==true){
//								pred_rank_4_DCG= convert_bucket(pred_rank);
//							}
//							//
//							writerDebug.append("\n 1.Top_N_max_for_calc_MAP:"+Top_N_max_for_calc_MAP+" pred_rank:"+pred_rank+" pred_rank_4_DCG:"+pred_rank_4_DCG
//													+" "+curr_q_topic+" "+ author_Name+" "+map_pred_rank_vs_GT_rank);
//							writerDebug.flush();
//							rel_i=pred_rank_4_DCG;
////							if(positionCount==1) rel_1=pred_rank_4_DCG;
//							map_i_relI.put( String.valueOf(i), rel_i);
//						}
						//if(isSOPprint)
						writerDebug.append("\ngt:"+map_pred_rank_vs_GT_rank.get(pred_rnk)+" pred:"+pred_rnk+" q_topic:"+last_q_topic+"\n");
						writerDebug.flush();
						// DCG preparation
						if(
//										map_pred_rank_vs_GT_rank.get(pred_rnk)<=Top_N_max_for_calc_MAP &&
														 pred_rnk<=Top_N_max_for_calc_MAP	
							){
							hit_counter++;
							//////////////////////DCG START							
							i=hit_counter;
							//
//							if(is_convert_using_bucket_for_rankingScore==false){
//								pred_rank_4_DCG=(Top_N_max_for_calc_MAP - map_pred_rank_vs_GT_rank.get(pred_rnk) )+1;
							
							if(is_use_original_ranking==true){
								if(is_convert_using_bucket_for_rankingScore==true){ //BUCKET
									int new_convertBucketRank=convert_bucket(map_pred_rank_vs_GT_rank.get(pred_rnk), writerDebug, true); //is_Debug
									pred_rank_4_DCG=new_convertBucketRank;
									writerDebug.append("\n bucket convert : pred_rank_4_DCG:"+pred_rank_4_DCG);
								}
								else{
									pred_rank_4_DCG= map_pred_rank_vs_GT_rank.get(pred_rnk); //actually GROUND TRUTH
								}
							}
							else if (is_use_original_ranking==false){
								pred_rank_4_DCG=(Top_N_max_for_calc_MAP - map_pred_rank_vs_GT_rank.get(pred_rnk) )+1;
							}
//								writerDebug.append("\n 2.Top_N_max_for_calc_MAP:"+Top_N_max_for_calc_MAP
//													+" GT:"+map_pred_rank_vs_GT_rank.get(pred_rnk)+" pred_rank_4_DCG:"+pred_rank_4_DCG
//													+" pred_rank_4_DCG:"+pred_rank_4_DCG+" "+curr_q_topic+" "+ author_Name+" "+map_pred_rank_vs_GT_rank);
//							}				//
//							else if(is_convert_using_bucket_for_rankingScore==true){
//								pred_rank_4_DCG= convert_bucket(pred_rnk, writerDebug, 
//																true); //is_Debug
//								//
//								writerDebug.append("\n 1.Top_N_max_for_calc_MAP:"+Top_N_max_for_calc_MAP+" pred_rnk:"+pred_rnk+" pred_rank_4_DCG:"+pred_rank_4_DCG
//														+" "+curr_q_topic+" "+ author_Name+" "+map_pred_rank_vs_GT_rank);
//							}
							writerDebug.flush();
							rel_i=pred_rank_4_DCG;
							writerDebug.append("\n bucket convert : rel_i"+rel_i);
							if(TEMP_MAX_NUMBER_AUTHOR<rel_i) TEMP_MAX_NUMBER_AUTHOR=rel_i;
							if(hit_counter==1){
								
								if(is_use_original_ranking){
									if(is_convert_using_bucket_for_rankingScore==true){ //BUCKET
//										int new_convertBucketRank=convert_bucket(pred_rank_4_DCG, writerDebug, false); //is_Debug
										rel_1_for_NUMERATOR=pred_rank_4_DCG;
									}
									else{ 
										rel_1_for_NUMERATOR=pred_rank_4_DCG; //ORIGINAL
									}
								}
								else{ // TRANSFORM
									//var 1
									int transformed_new_Predict_rank=(Top_N_max_for_calc_MAP-pred_rank_4_DCG)+1;
									rel_1_for_NUMERATOR=transformed_new_Predict_rank;
									//var 2
//									rel_1_for_NUMERATOR= (total_GT_rank-pred_rank_4_DCG)/total_GT_rank;
								}

							}
							writerDebug.append("\n adding to map_i_relI:"+" i="+i+" rel_i:"+rel_i);
							writerDebug.flush();
							map_i_relI.put(  i, rel_i);
							//////////////////////DCG END
							if(
									map_pred_rank_vs_GT_rank.get(pred_rnk)<=Top_N_max_for_calc_MAP &&
													 pred_rnk<=Top_N_max_for_calc_MAP	){
									hit_count++;							
									// 
									precision=hit_count/new Double(Top_N_max_for_calc_MAP);
									mean_precision=mean_precision+(hit_count/new Double(pred_rnk));
									
									System.out.println("HIT  PRED:"+pred_rnk+" GT:"+map_pred_rank_vs_GT_rank.get(pred_rnk)
														+" q_topic:"+last_q_topic+" hit_count:"+hit_count+" positionCount:"+positionCount
														+" precision:"+precision+" mean_precision:"+mean_precision
														+" Top_N_max_for_calc_MAP:"+Top_N_max_for_calc_MAP);
									if(hit_count==0 || pred_rnk==0 || String.valueOf(mean_precision).equalsIgnoreCase("nan") ){
										System.out.println("Error: NAN due to error of zero: hit_count="+hit_count+" pred_rnk:"+pred_rnk);
										writerDebug.append("\nError: NAN due to error of zero: hit_count="+hit_count+" pred_rnk:"+pred_rnk);
										writerDebug.flush();
									}
		
									writerDebug.append("\nHIT gt:"+map_pred_rank_vs_GT_rank.get(pred_rnk)+" pred:"+pred_rnk
												 		+" q_topic:"+last_q_topic+" hit_count:"+hit_count+" positionCount:"+positionCount
												 		+" precision:"+precision+" mean_precision:"+mean_precision +" Top_N_max_for_calc_MAP:"+Top_N_max_for_calc_MAP);
									writerDebug.flush();
							}
						}
					
					} // END for(int pred_rnk:map_pred_rank_vs_GT_rank.keySet()){
						//averaging IMP
						mean_precision=mean_precision/new Double(hit_count);
						/// 
						if(String.valueOf(mean_precision).equalsIgnoreCase("NaN") )
							mean_precision=0.;
						double sum_rel_by_logI=0.;
						double sum_rel_by_logI_for_IDCG=0.;
						//SORTING
						TreeMap< Integer, Integer> map_i_relI_SORTED_descend=new TreeMap< Integer, Integer>(); 
						writerDebug.append("\n map_i_relI:"+map_i_relI); writerDebug.flush();
						double IDCG_n=0.; int ideal_i_curr=0; double rel_1_for_DENOMINATOR=0.0;
						boolean is_empty_map_map_i_relI=true; int counter=0;
						MAX_NUMBER_AUTHORS=TEMP_MAX_NUMBER_AUTHOR; //replace with MAX AUTHORS from list (using GROUND TRUTH RANKING MAX VALUE)
						if(map_i_relI!=null){
							if(map_i_relI.size()>0  ){
								is_empty_map_map_i_relI=false;int new_convertBucketRank=-1;
								// DCG 
								for( int i_curr:map_i_relI.keySet()){
									counter++;
									///////////////  numerator  /////////
									//FLAG == 1
									if(Flag_choose_METHODtype_DCG==1){ // implements "Discounted Cumulative Gain" first equation in wikipedia
										if(Integer.valueOf(i_curr)>1){
//											sum_rel_by_logI+=map_i_relI.get(i_curr)
//															/ (double) LongMath.log2(Long.valueOf(i_curr) , RoundingMode.UNNECESSARY) ; //(   Math.log(i_curr)/ Math.log(2) );
											int  transformed_new_Predict_rank=MAX_NUMBER_AUTHORS-map_i_relI.get(i_curr) ;
											     transformed_new_Predict_rank=(Top_N_max_for_calc_MAP-map_i_relI.get(i_curr) )+1;
											// 
											if(is_use_original_ranking){
												if(is_convert_using_bucket_for_rankingScore==true){ //BUCKET
//													new_convertBucketRank=convert_bucket(map_i_relI.get(i_curr), writerDebug, false); //is_Debug
													sum_rel_by_logI+=map_i_relI.get(i_curr)
																	/  (   Math.log(Double.valueOf(i_curr))/ Math.log(2) ); //ORIGINAL
												}
												else{ // ORIGINAL
													sum_rel_by_logI+=map_i_relI.get(i_curr)
																	/  (   Math.log(Double.valueOf(i_curr))/ Math.log(2) ); //ORIGINAL
													
												}
											}
											else{ //TRANSFORMED
												//var 1 
												sum_rel_by_logI+=transformed_new_Predict_rank
																/  (   Math.log(Double.valueOf(i_curr))/ Math.log(2) );
												//var 2
//												sum_rel_by_logI+= ( (total_GT_rank-pred_rank_4_DCG)/total_GT_rank )
//																/  (   Math.log(Double.valueOf(i_curr))/ Math.log(2) );
												
											}
											 
											writerDebug.append("\n i_curr:"+i_curr+" map_i_relI.get(i_curr):"+map_i_relI.get(i_curr)
																+" bucket:"+new_convertBucketRank
//																+" log:"+LongMath.log2(Double.valueOf(i_curr) , RoundingMode.)
																+" log:"+String.valueOf((   Math.log(Double.valueOf(i_curr))/ Math.log(2) )).substring(0, 2)
//																+" partial:"+String.valueOf( map_i_relI.get(i_curr)/  (   Math.log(Double.valueOf(i_curr))/ Math.log(2) )).substring(0, 4)
																+" sum:"+String.valueOf(sum_rel_by_logI).substring(0, 2)
																	);
											writerDebug.flush();
										}
									}
									//FLAG == 2
									if(Flag_choose_METHODtype_DCG==2){// implements "Discounted Cumulative Gain" SECOND equation in wikipedia
//										sum_rel_by_logI+=  ((2^map_i_relI.get(i_curr)) - 1) / ( (double) LongMath.log2(Integer.valueOf(i_curr+1) , RoundingMode.HALF_DOWN)) ;
										sum_rel_by_logI+=  ((2^map_i_relI.get(i_curr)) - 1) / (   Math.log(Double.valueOf(i_curr+1))/ Math.log(2) );
									}
								} // END for( String i_curr:map_i_relI.keySet()){
								
								try{
									//FLAG == 1
									if(Flag_choose_METHODtype_DCG==1){ // implements "Discounted Cumulative Gain" first equation in wikipedia
//										DCG_n= map_i_relI.get("1") +sum_rel_by_logI;
										DCG_n= rel_1_for_NUMERATOR+sum_rel_by_logI;
									}
									//FLAG == 2
									if(Flag_choose_METHODtype_DCG==2){ // implements "Discounted Cumulative Gain" SECOND equation in wikipedia
										DCG_n=sum_rel_by_logI;
									}
								}
								catch(Exception e){
									DCG_n= 0.+sum_rel_by_logI;
									writerDebug.append("\n ERROR map_i_relI.get(1):"+map_i_relI.get("1")+" sum_rel_by_logI:"+sum_rel_by_logI);
									writerDebug.flush();
								}
								// Descending Sorting..
								//////////////////// IDCG 
//								map_i_relI_SORTED_descend= sort_given_treemap.sortByValue_SI_method2_descending(map_i_relI); // NOT WORKING
								map_i_relI_SORTED_descend_INT= sort_given_treemap.entriesSortedByValues(map_i_relI);
								Iterator<Entry<Integer, Integer>> iterator = map_i_relI_SORTED_descend_INT.iterator();
								int i_curr2=0;
								ideal_i_curr=0;
								while (iterator.hasNext()) {
									Entry<Integer, Integer> NOW=iterator.next();
//									System.out.println("NEXT:" + NOW.getKey() +" "+NOW.getValue());
//									map_i_relI_SORTED_descend.put(Integer.valueOf( NOW.getKey()), Integer.valueOf(NOW.getValue()));
									int curr_position= NOW.getKey(); int curr_rank=NOW.getValue();
									int transformed_new_Predict_rank=MAX_NUMBER_AUTHORS-curr_rank;
										transformed_new_Predict_rank=(Top_N_max_for_calc_MAP-curr_rank)+1;
									ideal_i_curr++;
									if(Integer.valueOf(ideal_i_curr)==1) {
										if(is_use_original_ranking){
											if(is_convert_using_bucket_for_rankingScore==true){ //BUCKET
//												new_convertBucketRank=convert_bucket(curr_rank, writerDebug, false); //is_Debug
												rel_1_for_DENOMINATOR=curr_rank;
											}
											else{
												rel_1_for_DENOMINATOR= curr_rank; // ORIGINAL												
											}
										}
										else{ //TRANSFORM
											//var 1
											rel_1_for_DENOMINATOR=transformed_new_Predict_rank;
											//var 2
//											rel_1_for_DENOMINATOR=(total_GT_rank-pred_rank_4_DCG)/total_GT_rank ; 
										}
									}
									if(Integer.valueOf(ideal_i_curr)>1){
//										sum_rel_by_logI_for_IDCG+=map_i_relI_SORTED_descend.get(i_curr)
//															/ (double) LongMath.log2(Integer.valueOf(ideal_i_curr) , RoundingMode.CEILING) ; //(   Math.log(i_curr)/ Math.log(2) );
										double log_denominator=( Math.log(Double.valueOf(ideal_i_curr))/ Math.log(2) );
										if(log_denominator==0){}
										else{
											if(is_use_original_ranking){
												if(is_convert_using_bucket_for_rankingScore==true){ //BUCKET
//													new_convertBucketRank=convert_bucket(curr_rank, writerDebug, false); //is_Debug
													sum_rel_by_logI_for_IDCG+=  curr_rank/log_denominator; 
												}
												else{
													sum_rel_by_logI_for_IDCG+=  curr_rank/log_denominator; //ORIGINAL
												}
											}
											else //TRANSFORM
												//var 1
												sum_rel_by_logI_for_IDCG+=  transformed_new_Predict_rank/log_denominator;
												//var 2
//												sum_rel_by_logI_for_IDCG+=  ((total_GT_rank-pred_rank_4_DCG)/total_GT_rank)/log_denominator;
										}
										
										writerDebug.append("\n ideal_i_curr:"+ideal_i_curr+" curr_rank:"+curr_rank
															+" bucket:"+new_convertBucketRank
					//										+" log:"+LongMath.log2(Double.valueOf(i_curr) , RoundingMode.)
															+" log:"+ String.valueOf((   Math.log(Double.valueOf(curr_position))/ Math.log(2) )).substring(0, 2)
															+" partial:"+ String.valueOf( curr_rank/log_denominator).substring(0, 2)
															+" sum:"+String.valueOf(sum_rel_by_logI_for_IDCG).substring(0, 2)
															+" curr_position:"+curr_position
																);
										
									}
								} //while (iterator.hasNext()) {
								
								IDCG_n = rel_1_for_DENOMINATOR + sum_rel_by_logI_for_IDCG;
								
								//////////////////// IDCG 
//								for( int i_curr:map_i_relI_SORTED_descend.keySet()){
//									ideal_i_curr++;
//									if(Integer.valueOf(ideal_i_curr)==1) {first_rel= map_i_relI_SORTED_descend.get(i_curr);}
//									//
//									if(Integer.valueOf(ideal_i_curr)>1){
////										sum_rel_by_logI_for_IDCG+=map_i_relI_SORTED_descend.get(i_curr)
////															/ (double) LongMath.log2(Integer.valueOf(ideal_i_curr) , RoundingMode.CEILING) ; //(   Math.log(i_curr)/ Math.log(2) );
//										sum_rel_by_logI_for_IDCG+=map_i_relI_SORTED_descend.get(i_curr)
//																   /( Math.log(Double.valueOf(i_curr))/ Math.log(2) );
//										
//										writerDebug.append("\n i_curr:"+i_curr+" map_i_relI_SORTED_descend.get(i_curr):"+map_i_relI_SORTED_descend.get(i_curr)
//					//										+" log:"+LongMath.log2(Double.valueOf(i_curr) , RoundingMode.)
//															+" log:"+(   Math.log(Double.valueOf(i_curr))/ Math.log(2) )
//															+" sum:"+sum_rel_by_logI
//																);
//										
//									}
//								}
//								IDCG_n=first_rel +sum_rel_by_logI_for_IDCG;
							} //END if(map_i_relI.size()>0  ){
							else{
							
								writerDebug.append("\n WARNING: map is EMPTY:" + curr_q_topic);
								writerDebug.flush();
								
							}
						} //END if(map_i_relI!=null){
						else{
							
							writerDebug.append("\n WARNING: map is NULL:" + curr_q_topic);
							writerDebug.flush();
							
						}
					 
						// DCG
						double nDCG_n=DCG_n / (double) IDCG_n;
						if(is_empty_map_map_i_relI==false)
							map_query_nDCG.put(last_q_topic, nDCG_n); //query-wise nDCG
						else
							map_query_nDCG.put(last_q_topic, 0.); //query-wise nDCG
						
						writerDebug.append("\n (num)DCG_n:"+DCG_n+" (denomintor)IDCG_n:"+IDCG_n
												+" nDCG_n:"+nDCG_n
												+" rel_1_for_NUMERATOR:"+rel_1_for_NUMERATOR
												+" rel_1_for_DENOMINATOR:"+rel_1_for_DENOMINATOR
												+" \n(num)sum_rel_by_logI:"+sum_rel_by_logI
												+" (den)sum_rel_by_logI_for_IDCG:"+sum_rel_by_logI_for_IDCG+"\n map_i_relI:"+map_i_relI
												+" map_i_relI_SORTED_descend:"+map_i_relI_SORTED_descend
												+" map_i_relI_SORTED_descend_INT:"+map_i_relI_SORTED_descend_INT
												+" for query:"+curr_q_topic
												+" map_query_nDCG:"+map_query_nDCG);
						writerDebug.flush();
						
					System.out.println("FINAL HIT gt:"
										+" q_topic:"+last_q_topic+" hit_count:"+hit_count+" positionCount:"+positionCount
										+" precision:"+precision+" mean_precision:"+mean_precision
										+" map_pred_rank_vs_GT_rank.size:"+map_pred_rank_vs_GT_rank.size());
					writerDebug.append("\nFINAL HIT gt:"+" q_topic:"+last_q_topic+" hit_count:"+hit_count+" positionCount:"+positionCount
										+" precision:"+precision+" mean_precision:"+mean_precision
										+" map_pred_rank_vs_GT_rank.size:"+map_pred_rank_vs_GT_rank.size()
										+" map_qTopicID_Mean_Precision:"+map_qTopicID_Mean_Precision
										+" last_q_topic:"+last_q_topic+" curr_q_topic:"+curr_q_topic);
					writerDebug.flush();
					
					//  LAST TOPIC
					if(lineNumber==lineNumber2){
						writerDebug.append("\n LAST TOPIC :"+curr_q_topic+" precision:"+precision+" mean_precision:"+mean_precision
												+" map_qTopicID_Precision:" +map_qTopicID_Precision+ " map_qTopicID_Mean_Precision:"+map_qTopicID_Mean_Precision);
						writerDebug.flush();
						
						//calc precision
						map_qTopicID_Precision.put(curr_q_topic, precision);
						 // mean precision
						map_qTopicID_Mean_Precision.put(curr_q_topic, mean_precision);						
					}
					else {
						//calc precision
						map_qTopicID_Precision.put(last_q_topic, precision);
						 // mean precision
						map_qTopicID_Mean_Precision.put(last_q_topic, mean_precision);
					}
					
					System.out.println("precision:"+precision+
									  " mean precision:"+mean_precision+
									  " positionCount:"+positionCount+
									  " hit_count:"+hit_count+" for "+last_q_topic
									  );
					writerDebug.append("\nprecision:"+precision+" mean precision:"+mean_precision+" positionCount:"+positionCount+
							  		" hit_count:"+hit_count+" for "+last_q_topic);
					writerDebug.flush();
					
					mean_precision=0;
					
					map_GT_rank_vs_pred_rank=new TreeMap<Integer, Integer>();
					map_pred_rank_vs_GT_rank=new TreeMap<Integer, Integer>();
					
					map_i_relI=new TreeMap<Integer, Integer>();
					System.out.println("reset MAP");
				} // END if(!curr_q_topic.equalsIgnoreCase(last_q_topic) && last_q_topic.length()>0){
//				else{
//					// very LAST topic 
//					//calc precision
//					map_qTopicID_Precision.put(last_q_topic, precision);
//					 // mean precision
//					map_qTopicID_Mean_Precision.put(last_q_topic, mean_precision);
//					
//				}
				
				
				// ITERATE the last query where 
				
				reset_counter++;
//				map_GT_rank_vs_pred_rank.put(GT_rank, reset_counter);
//				map_pred_rank_vs_GT_rank.put(reset_counter, GT_rank);
				
				map_GT_rank_vs_pred_rank.put(GT_rank, pred_rank);
				map_pred_rank_vs_GT_rank.put(pred_rank, GT_rank);
				
				//System.out.println(token);
				if(isSOPprint)
					System.out.println(token+" 1:"+curr_q_topic
											+" 2:"+ GT_rank
											+" series:"+reset_counter
									  );
				
				writerDebug.append("\n "+"curr_q_topic:"+curr_q_topic+" map_GT_rank_vs_pred_rank:"+map_GT_rank_vs_pred_rank);
				writerDebug.flush();

				last_q_topic=token.substring(0, token.indexOf("("));
			} //end while read of inFile_output_matched
			int hit_count=0;
			mean_precision=0;
			positionCount=0;
			//LAST ONE adding (***** BELOW COMMENTED as this has been handled above using lineNumber==lineNumber2 condition)
//			for(int pred_rnk:map_pred_rank_vs_GT_rank.keySet()){
//				positionCount++;
// 
//				writerDebug.append("\nGTrank:"+map_pred_rank_vs_GT_rank.get(pred_rnk)+" pred_rank:"+pred_rnk+" q_topic:"+last_q_topic+"\n");
//				writerDebug.flush();
//				//
//				if(map_pred_rank_vs_GT_rank.get(pred_rnk)<=Top_N_max_for_calc_MAP &&
//						pred_rnk<=Top_N_max_for_calc_MAP	){
//					hit_count++;
//					precision=hit_count/new Double(new Double(Top_N_max_for_calc_MAP));
//					mean_precision=mean_precision+(hit_count/new Double(pred_rnk)); //positionCount
//					System.out.println("GTrank:"+map_pred_rank_vs_GT_rank.get(pred_rnk)+" pred_rank:"+pred_rnk+" q_topic:"+last_q_topic +" name:");
//					writerDebug.append("\nGTrank:"+map_pred_rank_vs_GT_rank.get(pred_rnk)+" pred_rank:"+pred_rnk+" q_topic:"+last_q_topic);
//					writerDebug.flush();
//				}
//				map_qTopicID_Precision.put(last_q_topic, precision);
//			}
//			mean_precision=mean_precision/new Double(hit_count);
//			// mean precision
//			map_qTopicID_Mean_Precision.put(last_q_topic, mean_precision);
			//
			int cnt20=1;
			avg_precision=0.0;
			for(String curr_q_topic_1:map_currQtopic_GT_rank_vs_pred_rank.keySet()){
				System.out.println("precision-> q topic:"+curr_q_topic_1+" MAP:"+map_qTopicID_Precision.get(curr_q_topic_1)+" count:"+cnt20+" for top N->"+Top_N_max_for_calc_MAP);
				writerDebug.append("\nprecision-> q topic:"+curr_q_topic_1+" MAP:"+map_qTopicID_Precision.get(curr_q_topic_1)+" count:"+cnt20+" for top N->"+Top_N_max_for_calc_MAP);
				writer.append("\nprecision-> q topic:"+curr_q_topic_1+" MAP:"+map_qTopicID_Precision.get(curr_q_topic_1)+" count:"+cnt20+" for top N->"+Top_N_max_for_calc_MAP);
				avg_precision=avg_precision+map_qTopicID_Precision.get(curr_q_topic_1);
				cnt20++;
			}
			
			writerDebug.append("\n map_qTopicID_Mean_Precision:"+map_qTopicID_Mean_Precision+" last_q_topic:"+last_q_topic+" curr_q_topic:"+curr_q_topic); 
			writerDebug.flush();
			mean_avg_precision=0.0; cnt20=1;
			// 
			for(String curr_q_topic2:map_qTopicID_Mean_Precision.keySet()){
				System.out.println("mean precision-> q topic:"+curr_q_topic2+" MAP:"+map_qTopicID_Mean_Precision.get(curr_q_topic2)+" count:"+cnt20+" for top N->"+Top_N_max_for_calc_MAP);
				writerDebug.append("\nmean precision-> q topic:"+curr_q_topic2+" MAP:"+map_qTopicID_Mean_Precision.get(curr_q_topic2)+" count:"+cnt20+" for top N->"+Top_N_max_for_calc_MAP);
				writer.append("\nmean precision-> q topic:"+curr_q_topic2+" MAP:"+map_qTopicID_Mean_Precision.get(curr_q_topic2)+" count:"+cnt20+" for top N->"+Top_N_max_for_calc_MAP);
				
				mean_avg_precision=mean_avg_precision+map_qTopicID_Mean_Precision.get(curr_q_topic2);
				cnt20++;
			}
			double sum_nDCG_n=0.;
			//fill the missing curr query with zero (ideally should not be required)
			for(String cQuery2:map_UNIQUE_query_from_inFile_as_KEY.keySet()){
				if(!map_query_nDCG.containsKey(cQuery2))
					map_query_nDCG.put(cQuery2,0.);
			}
			double new_sum_nDCG_n=0.;
			// DCG (final)
			if(is_use_original_ranking==true){
				for(String cQuery:map_query_nDCG.keySet()){
					sum_nDCG_n+=map_query_nDCG.get(cQuery);
				}
				new_sum_nDCG_n=sum_nDCG_n;
			}
			else{ //TRANSFORMTION
				for(String cQuery:map_query_nDCG.keySet()){
					sum_nDCG_n+=map_query_nDCG.get(cQuery);
				}
				// TRANSFORMATION (NORMALIZE EACH VALUE
//				for(String cQuery:map_query_nDCG.keySet()){
//					double curr_value= map_query_nDCG.get(cQuery) / (double)sum_nDCG_n;
//					map_query_nDCG.put(cQuery, curr_value); //reinsert NORMALIZED VALUE
//				}
				//new
//				for(String cQuery:map_query_nDCG.keySet()){
//					new_sum_nDCG_n+=map_query_nDCG.get(cQuery);
//				}
				new_sum_nDCG_n=sum_nDCG_n;
			}
			
			System.out.println("****map_currQtopic_GT_rank_vs_pred_rank.size(CHECK if this matches):"+map_currQtopic_GT_rank_vs_pred_rank.size());
			writerDebug.append("\n****map_currQtopic_GT_rank_vs_pred_rank.size(CHECK if this matches):"+map_currQtopic_GT_rank_vs_pred_rank.size());
			writer.append("\n****map_currQtopic_GT_rank_vs_pred_rank.size(CHECK if this matches):"+map_currQtopic_GT_rank_vs_pred_rank.size());
			
			
			System.out.println("avg_precision:"+avg_precision/new Double(map_qTopicID_Precision.size())
													+" of size="+map_qTopicID_Precision.size());
			writerDebug.append("\navg_precision:"+avg_precision/new Double(map_qTopicID_Precision.size())
							+" of size="+map_qTopicID_Precision.size() +" for top N->"+Top_N_max_for_calc_MAP +"\n");
			writer.append("\navg_precision:"+avg_precision/new Double(map_qTopicID_Precision.size())
							+" of size="+map_qTopicID_Precision.size()+" for top N->"+Top_N_max_for_calc_MAP +"\n");

			
			System.out.println("mean_avg_precision:"+mean_avg_precision/new Double(map_qTopicID_Mean_Precision.size())
													+" of size="+map_qTopicID_Mean_Precision.size()+" for top N->"+Top_N_max_for_calc_MAP );
			writerDebug.append("\nmean_avg_precision:"+mean_avg_precision/new Double(map_qTopicID_Mean_Precision.size())
													+" of size="+map_qTopicID_Mean_Precision.size()+" for top N->"+Top_N_max_for_calc_MAP );
			writer.append("\nmean_avg_precision:"+mean_avg_precision/new Double(map_qTopicID_Mean_Precision.size())
							+" of size="+map_qTopicID_Mean_Precision.size()+" for top N->"+Top_N_max_for_calc_MAP );
			writer.append("\n --->nDCG_n:"+new_sum_nDCG_n / (double) map_query_nDCG.size() +" map_query_nDCG:"+map_query_nDCG+" map_query_nDCG.size():"+map_query_nDCG.size()
								+" sum_nDCG_n:"+new_sum_nDCG_n);
			System.out.println("# nDCG_n:"+new_sum_nDCG_n / (double) map_query_nDCG.size() +" map_query_nDCG:"+map_query_nDCG);
			
			// OUT MAP
			mapOut.put("precision", avg_precision/new Double(map_qTopicID_Precision.size()));
			mapOut.put("map", mean_avg_precision/new Double(map_qTopicID_Mean_Precision.size()) );
			mapOut.put("dcg", new_sum_nDCG_n / (double) map_query_nDCG.size() );
		
			System.out.println("GIVEN inFile_output_matched:"+inFile_output_matched);
			
			writerDebug.append("\nGIVEN inFile_output_matched:"+inFile_output_matched);
			writerDebug.append("\n------------------END-------------------------\n");
			
			writerDebug.flush();
			writer.flush();
		}
		catch(Exception e){
			e.printStackTrace();
		}
		return mapOut;
	}
	// copy_Top_N_files_to_runX_folder
	private static void copy_Top_N_files_to_runX_folder(
								 String[] arr_top_N_files, 
								 String base_prefix_folder
								,String base_LDA_folder
								) {
		// TODO Auto-generated method stub
		try {
			int c=0;
			String dest_="";
			while(c<arr_top_N_files.length){
				
				String source_=arr_top_N_files[c];
				System.out.println("arr_top_N_files[c]:"+c +" total:"+arr_top_N_files.length);
				
				String source_filename=new File(arr_top_N_files[c]).getName();
				
				if(source_filename.indexOf("LDA_") ==-1)
					dest_=base_prefix_folder+source_filename;
				else
					dest_=base_LDA_folder+source_filename;
				
				
				File source_file=new File(source_);
				File dest_file=new File(dest_);
				
				
				
				System.out.println("source_file:"+source_file+" dest_file:"+dest_file);
				FileUtils.copyFile(source_file, dest_file);
				c++;
			}
			
		} catch (Exception e) {
			e.printStackTrace();
			// TODO: handle exception
		}
		
	}	
	//main
	public static void main(String[] args) throws IOException {
			long t0 = System.nanoTime();
			
			//Pre-Requisiton 
			// (0) run "p6_MAIN_Work.java" before running this method.
			// (1) run via "p6_MAIN_Work.java" tf_idf_AND_cosine_main() for inFile_1_tf_idf_each_doc (if not exist already) 
			// (2) run around 3 methods in main() of "find_VERB_AND_Noun_JustBrf_from_TaggedString_inFile()"
			//		that will give file of   "output_organiz_person_locations_numbers_docID.txt", and other files
			// (3) variable "CSV_containing_10_interested_topics" is correctly SET?
			// (4) variable "latent_topic_manuallyFound_CSV" is correctly SET?
			// variables folder_prefix_LDA_or_PLSI, baseFolder_PLSI_or_LDA is set CORRECT..
			// (5) flag_run_method==3 --> run LDA from mallet or PLSI
			// (6) flag_run_method==3 ---> Use p6_ground_truth_plsi() method to convert a file having "d z probability" 
			//		to a file in var "lda_R_hlda_R_plsi_Top_N" for each author.
			// 	flag==3 run  will produce "LDA_outFile_each_auth_CUMM_topic_score.txt"
			
			String baseFolder="/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/";
			// run tf_idf_AND_cosine_main() for inFile_1_tf_idf_each_doc
		    String inFile_1_tf_idf_each_doc=baseFolder+"doc_id_word_id_AND_tfidf__.txt";// can be freq or tf-idf
		    // ****** run around 3 methods in main() of "find_Verb_AND_Noun_JustBfr_from_TaggedString_inFile" to get 
		    //       the "inFile_2_Distinct_Noun_eachLine_as_doc" BELOW
		    String inFile_2_Distinct_Noun_eachLine_as_doc=baseFolder+"Noun_count.txt";
		    //****** run around 3 methods in main()   of "find_Verb_AND_Noun_JustBfr_from_TaggedString_inFile" *******************
		    String inFile_8_get_organization_person_location_docID=baseFolder+"output_organiz_person_locations_numbers_docID.txt";
		    // inFile_11_* <- NOT yet implemented...
		    String inFile_11_repository_of_organization_person_location_docID_pastRan=baseFolder+"";
		    //(to be removed)This file from output of main() of find_Verb_AND_Noun_JustBfr_from_TaggedString_inFile() after Title NLP
		    //******This file is output FROM Flag==5 run of "cleaning_and_convert"
		    String inFile_3_with_all_tokens=baseFolder+"20000.txt";		    											
		    int    token_no_contain_authors_in_inFile_3=4; //change manual
		    int    token_no_contain_bodyText_in_inFile_3=8; //change manual
			// flag_run_method==2   ---- START
			int index_of_token_of_Source_FileName=7 ; 		//change manual
			int index_of_token_of_SourceURL=2; 				//change manual
			int index_of_token_interested_for_freq_count=token_no_contain_authors_in_inFile_3; //change manual
			// flag_run_method==2   ---- END
		    
		    //**NOTE: this below file is NOT used in calc_ground_truth()
		    String inFile_4_contains_only_bodyText=baseFolder+"mergedall-2016-T9-10-Topics-all-tokens.txt_ONLY_ENGLISH.txt_removed_NOISE.txt_REMOVE_STOPWORDS_FINAL.txt"; //obsolete
		    String inFile_5_AuthorID_Doc_ID_matrix=baseFolder+"auth_id_doc_id.txt";
		    String inFile_6_AuthorName_AND_Auth_ID=baseFolder+"authName_AND_auth_id.txt";
		    String inFile_7_auth_id_doc_id_queryTopicRelated=baseFolder+"auth_id_doc_id_queryTopicRelated.txt"; // approach_2
		    String inFile_9_auth_id_doc_id_queryTopicRelated__applying_multi_crossAuthIDs
		    									  =baseFolder+"auth_id_doc_id_queryTopicRelated.txt__applying_multi_crossAuthIDs.txt";
		    
		    String inFile_10_authName_AND_auth_id__MAPPEDmultiAuthID_SORTED_add_minAuthID
		    									  =baseFolder+"authName_AND_auth_id.txt__MAPPEDmultiAuthID_SORTED_add_minAuthID.txt";
		    String  inFile_11_having_DIRTY_author_name=baseFolder+"authName_AND_auth_id.txt_YES_dirtyAuthNames.txt"; //DIRTY AUTHNAME, AUTHID
		    String  inFile_12_having_NO_DIRTY_author_name=baseFolder+"authName_AND_auth_id.txt_NO_dirtyAuthNames.txt"; //DIRTY AUTHNAME, AUTHID
		    
		    String inputFolderName_inFile_3=baseFolder+""; //obsolete
		    String outputFileName_inFile_3_keyword_AND_freq=baseFolder+"authName_AND_freq.txt";
		    String outputFileName_onlykeywords_inFile_3=baseFolder+"onlyAuthorName.txt";
		    String debugFile=baseFolder+"debug_gt.txt";
		    String output_Ground_Truth=baseFolder+"Ground_Truth_OUT_auth_id_CUMM_SUM_AVG_tf_idf.txt";
		    String output_Ground_Truth_removeDuplicate=baseFolder+"Ground_Truth_OUT_auth_id_CUMM_SUM_AVG_tf_idf_noDup.txt";
		    
		    //{1,2} , 2- works faster, needs inFile_7_auth_id_doc_id_queryTopicRelated
		    int     Flag_2_choose_approach_topicDoc_count_for_each_author=2;//***
		    boolean is_skip_calling_calc_Frequency_Of_Token_AND_output_unique_values_in_token=true; //skip
		    String 	pattern_for_sourceChannel_TRAD="trad_";
		    String 	pattern_for_sourceChannel_POLI="poli_";
		    
		    //--------- BEGIN flag==1
		    
		    	/////// below setting for processing LDA output <d z prob> from mallet file
		        String folder_prefix_LDA_or_PLSI="LDA";
		        String baseFolder_PLSI_or_LDA=baseFolder+ folder_prefix_LDA_or_PLSI+ "/"+folder_prefix_LDA_or_PLSI+"_run1/";
		        // INPUT
		        //below var USUALLY "LDA_d_by_z_prob.txt"
		        String inFile_1_output_file_plsi=baseFolder_PLSI_or_LDA+"LDA_d_by_z_prob.txt"; //INPUT
		        String inFile_2_authorID_docIDCSV=baseFolder+"auth_id_doc_id.txt_NO_dirtyAuthNames.txt"; //INPUT
		        String inFile_3_authorName_AuthID=baseFolder+"authName_AND_auth_id.txt_NO_dirtyAuthNames.txt"; //INPUT
				String replace_string_for_sourceChannelFilePath="infile:/users/lenin/downloads/#problems/p6/merged_/rawxmlfeeds2csv/merged/all_all_body_url_datetime_auth_keywor_subjec_";
				
		        //OUTPUT file
		        String outFile_each_auth_CUMM_topic_score=baseFolder_PLSI_or_LDA+ folder_prefix_LDA_or_PLSI+ "_outFile_each_auth_CUMM_topic_score.txt";
		        String outFile_each_auth_CUMM_topic_score_POLI=baseFolder_PLSI_or_LDA+ folder_prefix_LDA_or_PLSI+ "_outFile_each_auth_CUMM_topic_score_POLI.txt";
		        String outFile_each_auth_CUMM_topic_score_TRAD=baseFolder_PLSI_or_LDA+ folder_prefix_LDA_or_PLSI+ "_outFile_each_auth_CUMM_topic_score_TRAD.txt";
		 
		        //from LDA run latent topic, z=0 to 9 (manually marked for each clusters from z=0 to 9)
		        String latent_topic_manuallyFound_CSV="india,boko haram,syria,climate change,crime,election,britain";
		        	   latent_topic_manuallyFound_CSV="crime,india,boko haram,syria,election,climate change";
		        	   latent_topic_manuallyFound_CSV="election,climate change,syria,boko haram,india,crime"; // LDA
		        	   latent_topic_manuallyFound_CSV="india,election,syria, boko haram,climate change,crime"; //LDA
		        	   latent_topic_manuallyFound_CSV="election,boko haram,syria,crime,india,climate change";
		        	   latent_topic_manuallyFound_CSV="india,election,climate change,syria,crime,boko haram";
		        	   latent_topic_manuallyFound_CSV="climate change,syria,india,boko haram,crime,election";
		        	   latent_topic_manuallyFound_CSV="climate change,syria,boko haram,india,crime,election";
		        	   latent_topic_manuallyFound_CSV="india,boko haram,crime,climate change,syria,election";
		        	   latent_topic_manuallyFound_CSV="india,boko haram,crime,climate change,syria,election";
		        	   latent_topic_manuallyFound_CSV="boko haram,syria,india,climate change,election,crime";
		        	   latent_topic_manuallyFound_CSV="crime,syria,india,climate change,boko haram,election";
		        	   latent_topic_manuallyFound_CSV="climate change,india,election,boko haram,crime,state";
		        	   latent_topic_manuallyFound_CSV="crime,boko haram,climate change,election,syria,india";
		        	   latent_topic_manuallyFound_CSV="climate change,india,crime,syria, boko haram,election";
		        	   latent_topic_manuallyFound_CSV="election,india,syria,climate change,boko haram,crime";
		        	   latent_topic_manuallyFound_CSV="india,crime,boko haram,climate change,syria,election";
		        	   latent_topic_manuallyFound_CSV="climate change,crime,election,india,boko haram,syria";
		        	   latent_topic_manuallyFound_CSV="syria,boko haram,crime,election,india,climate change";
		        	   latent_topic_manuallyFound_CSV="india,climate change,boko haram,syria,election,crime";
		        	   latent_topic_manuallyFound_CSV="boko haram,climate change,syria,india,crime,election";
		        	   latent_topic_manuallyFound_CSV="syria,boko haram,india,election,climate change,crime";
		        	   latent_topic_manuallyFound_CSV="election,boko haram,syria,india,crime,climate change";
		        	   latent_topic_manuallyFound_CSV="boko haram,syria,climate change,crime,india,election";
		        	   latent_topic_manuallyFound_CSV="election,syria,boko haram,climate change,india,crime";
		        	   latent_topic_manuallyFound_CSV="election,india,climate change,crime,syria,boko haram";
		        	   latent_topic_manuallyFound_CSV="climate change,election,boko haram,syria,india,crime";
		        	   latent_topic_manuallyFound_CSV="climate change,india,election,boko haram,syria,crime";
		        	   latent_topic_manuallyFound_CSV="crime,election,india,boko haram,syria,climate change";
//		        	   latent_topic_manuallyFound_CSV="election,climate change,boko haram,crime,syria,india"; //equ3 (2ndAttempt) another try (no use)
		        	   latent_topic_manuallyFound_CSV="climate change,election,crime,boko haram,india,syria";
		        	   
		        	   latent_topic_manuallyFound_CSV="boko haram,syria,climate change,election,india,crime"; //SimpleLDA (2ndAttempt.ALL_files2)
		        	   latent_topic_manuallyFound_CSV="crime,climate change,boko haram,election,syria,india"; //equ3 (2ndAttempt)
		        	   latent_topic_manuallyFound_CSV="climate change,syria,election,crime,india,boko haram"; //equ2 (2ndAttempt)
		        	   latent_topic_manuallyFound_CSV="india,crime,climate change,boko haram,election,syria"; //equ1 (2ndAttempt)
		        	   
		        	   latent_topic_manuallyFound_CSV="boko haram,syria,india,climate change,crime,election"; //equ1 (3rdAttempt)
		        	   latent_topic_manuallyFound_CSV="boko haram,india,election,crime,syria,climate change"; //equ3 (3rdAttempt)
		        	   latent_topic_manuallyFound_CSV="crime,india,climate change,election,syria,boko haram"; //equ2 (3rdAttempt)*****another try
		        	   latent_topic_manuallyFound_CSV="india,crime,climate change,election,syria,boko haram"; //equ2 (3rdAttempt)
		        	   latent_topic_manuallyFound_CSV="india,climate change,crime,syria,boko haram,election"; //SimpleLDA (3rdAttempt)
		        	   latent_topic_manuallyFound_CSV="india,boko haram,climate change,election,syria,crime"; //equ5 (attempt1)
		        	   latent_topic_manuallyFound_CSV="india,election,crime,syria,climate change,boko haram"; //equ5 (attempt3)
		        	   latent_topic_manuallyFound_CSV="india,election,syria,crime,boko haram,climate change"; //equ5 (attempt2)
		        	   latent_topic_manuallyFound_CSV="india,crime,boko haram,syria,climate change,election"; //equ5 (attempt4)
		        	   latent_topic_manuallyFound_CSV="boko haram,india,climate change,crime,election,syria"; //equ5 (attempt5)
		        	   latent_topic_manuallyFound_CSV="boko haram,crime,climate change,crime,india,syria"; //equ5 (attempt5) --another try (bad)
		        	   latent_topic_manuallyFound_CSV="india,crime,climate change,election,boko haram,syria";//equ5 (attempt6)
		        	   latent_topic_manuallyFound_CSV="boko haram,india,election,boko haram,climate change,syria"; //equ5 (attempt7)
		        	   latent_topic_manuallyFound_CSV="crime,india,election,syria,climate change,boko haram"; //equ5 (attempt8)
		        	   latent_topic_manuallyFound_CSV="climate change,boko haram,syria,india,crime,election"; //PLSI (attempt1)
		        	   latent_topic_manuallyFound_CSV="crime,climate change,india,election,syria,boko haram"; //PLSI (attempt2)
		        	   latent_topic_manuallyFound_CSV="syria,election,climate change,crime,boko haram,india"; //PLSI (attempt3)
		        	   latent_topic_manuallyFound_CSV="syria,election,india,crime,boko haram,climate change"; //PLSI attempt 4
		        	   latent_topic_manuallyFound_CSV="india,syria,boko haram,climate change,crime,election";  //equ5 attempt3 (id=12344)"
		        	   latent_topic_manuallyFound_CSV="crime,boko haram,syria,election,india,climate change";//equ5 attempt3 (id=12345)"
		        	   latent_topic_manuallyFound_CSV="climate change,syria,india,election,crime,boko haram";//equ5 attempt3 (id=12346)"
		        	   latent_topic_manuallyFound_CSV="boko haram,crime,india,election,climate change,syria";//equ5 attempt3 (id=12347)"
		        	   latent_topic_manuallyFound_CSV="boko haram,india,syria,climate change,election,crime";//equ5 attempt3 (id=12348)"
		        	   latent_topic_manuallyFound_CSV="boko haram,election,syria,india,climate change,crime";//equ5 attempt3 (id=12349)"
		        	   latent_topic_manuallyFound_CSV="india,boko haram,climate change,election,syria,crime";//equ5 attempt3 (id=12350)"
		        	   latent_topic_manuallyFound_CSV="boko haram,syria,india,climate change,election,crime";//equ5 attempt3 (id=12351)"
		        	   
		        	   latent_topic_manuallyFound_CSV="boko haram,crime,election,india,syria,climate change";//equ6 attempt1 (id=22350)"
		        	   latent_topic_manuallyFound_CSV="climate change,crime,india,boko haram,election,syria";//equ6 attempt2 (id=22351)"
		        	   latent_topic_manuallyFound_CSV="climate change,india,election,boko haram,crime,syria";//equ6 attempt1 (id=22352)
//		        	   latent_topic_manuallyFound_CSV="crime,boko haram,climate change,syria,india,election";//equ7 attempt1 (id=32350)
		        	   latent_topic_manuallyFound_CSV="crime,climate change,india,boko haram,syria,election"; //equ6 attempt4 (id=22354)
		        	   latent_topic_manuallyFound_CSV="syria,india,crime,climate change,boko haram,election"; //equ6 attempt5 (id=22355)
		        	   latent_topic_manuallyFound_CSV="boko haram,election,india,climate change,crime,syria";//equ7 attempt_2 (id=32351)
		        	   latent_topic_manuallyFound_CSV="boko haram,election,climate change,crime,india,syria"; // PAM output
		        	   
		    // this is NOT from LDA, this is general all the topics we have 
		   	String CSV_containing_10_interested_topics="india,syria,boko haram,climate change,crime,election";
				  //cancer,traffick
		    //--------- END flag==1
			//--------- START flag==3
		   	// SOME AUTH ID found in PRED NOT found in Ground truth
		   	String outputFile_missing_AUTHID_while_MATCHING_from_Flag_3=baseFolder+"missing_AUTH_ID_on_match_PRED_vs_GT.txt"; // DONT NEED CHANGE
		    int GT_int=8;  //p6-best{3,1,7};worst-{4,5,6}
		   	//--------- END flag==3
			//--------- START flag==4		   
		    String output_Result=baseFolder+"out_results_MAP.txt";
			int top_N_match=50; //Flag==4
		    //--------- END flag==4
		    //flag==3->(now automated, NOT NEEDED)copy the output files from  "Ground_Truth_final_x_OUT.txt", "Ground_Truth_OUT_auth_id_CUMM_SUM_AVG_tf_idf.txt","Ground_Truth_final_5_authName_freq_OUT.txt" 
		    // "tf_idf_sum_avg.txt" from baseFolder to folder mentioned in var "base_groundTruth_folder"
		    int flag_run_method=34;  		    // *****FLAG (=34 run after running LDA)
		    int Flag_choose_METHODtype_DCG=1; //2 different formula exists  // MANUAL CHANGE
		    //"is_use_original_ranking" has to be TRUE if this is true; // MANUAL CHANGE
		    boolean is_convert_using_bucket_for_rankingScore=true; //<--this works TRUE only if "is_use_original_ranking==true"  
		    boolean is_use_original_ranking=true;//set this FALSE to use TRANFORMATION using MAX_NUMBER_AUTHOR
		    //**************** Flag_for_Task ****************************************************************
            // 1  = calc_ground_truth()
            // 2  = calc_ground_truth_2()
            // 3  = calc_ground_truth_compare_LDA_OR_hlda_plsi()
            // 4  = calc_mean_avg_precision_MAP()
		    // 99 = run all the Flags of them in sequence [1..4]
		    // 12 = run the Flags in the sequence [1,2]
		    // 34 = run the Flags in the sequence [3,4]
            //************************************************************************************************
		    
		     ///////////////////////////////////////////////////////////////////////////////////////
		    if(flag_run_method==1 || flag_run_method==99 || flag_run_method==12 ){
		    
		      // 6 minutes for approach 2 (Flag_2_choose_approach_topicDoc_count_for_each_author=2)
			 //TIME: 8 hours for 7800 authors for approach 1, 
		    // CALCULATE GROUND TRUTH
		    calc_ground_truth(
					    		  baseFolder, // baseFolder
							      inFile_1_tf_idf_each_doc,
							      inFile_2_Distinct_Noun_eachLine_as_doc,
							      inFile_3_with_all_tokens,
							      token_no_contain_authors_in_inFile_3,
							      token_no_contain_bodyText_in_inFile_3,
							      inFile_4_contains_only_bodyText,
							      inFile_5_AuthorID_Doc_ID_matrix,
							      inFile_6_AuthorName_AND_Auth_ID,
							      inFile_7_auth_id_doc_id_queryTopicRelated,
							      inFile_8_get_organization_person_location_docID,
							      inFile_10_authName_AND_auth_id__MAPPEDmultiAuthID_SORTED_add_minAuthID,
							      inputFolderName_inFile_3, //obsolete
							      outputFileName_inFile_3_keyword_AND_freq, //author name and freq //OUTPUT
							      outputFileName_onlykeywords_inFile_3, 	//author name only//OUTPUT
							      CSV_containing_10_interested_topics,
							      output_Ground_Truth, //OUT
							      output_Ground_Truth_removeDuplicate, // OUT
							      Flag_2_choose_approach_topicDoc_count_for_each_author,// {1,2} , 2- works faster, needs inFile_7_auth_id_doc_id_queryTopicRelated
							      is_skip_calling_calc_Frequency_Of_Token_AND_output_unique_values_in_token,
							      replace_string_for_sourceChannelFilePath,
							      debugFile
								);
		    
		     System.out.println("output_Ground_Truth:"+output_Ground_Truth);
		     
		     //HERE process the output from LDA  mallet run. 
		     System.out.println("NOTE: you might have to correct MIN AUTH ID file (lines at the end - remove) to make sure some auth ID missing later at the Flag==3 or 4 stage..."
		     						+ " Some Auth ID in PREDICTED Rank is NOT in Ground truth for Flag==3..");
		     System.out.println("---flag_run_method==1 COMPLETED in ground_truth...."); 
		    }
		    
  		    //STEP  this file output from calc_ground_truth
		    String inFile_outputFileName_inFile_3_keyword_AND_freq_noduplicates=baseFolder+		    		
		    													   "outputUniqNames_auth_DistinctDocs_no_duplicates.txt";
		    
			String outputFileName="";
			String outputFileName_onlykeywords="";
			//only keywords
			outputFileName=baseFolder+"uniq_authors.txt";
			outputFileName_onlykeywords=baseFolder+"OnlyAuthors_asc_length.txt";
 
					debugFile=baseFolder+"debugTooken.txt";
			String  delimiter_4_inputFolderName_OR_Single_File="!!!";
			boolean is_split_on_blank_space_in_token=false;
			boolean is_do_stemming_on_word=false;
			boolean is_split_CSV_on_the_given_token=true;
			boolean is_token_interested__authors=true;

			// 
		    if(flag_run_method==2|| flag_run_method==99 || flag_run_method==12
		    		|| flag_run_method==234){
		    	
		    	//METHOD: calc_Frequency_Of_Token_AND_output_unique_values_in_token ( comment/uncomment as req)
		    	readFile_eachLine_get_a_particular_token_calc_frequency.
				calc_Frequency_Of_Token_AND_output_unique_values_in_token
																	(
																	inFile_3_with_all_tokens, //inputFolderName_OR_Single_File
																	delimiter_4_inputFolderName_OR_Single_File,
																	outputFileName, // OUTPUT
																	false, //is_Append_outputFileName,
																	outputFileName_onlykeywords,
																	false, //is_Append_outputFileName_onlykeywords,
																	outputFileName+"_WORD_DOCUMENT_FREQ.txt",
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
				   

			       String outFile_noDuplicates=baseFolder+"outputUniqNames_auth_DistinctDocs_no_duplicates.txt";
			       //This file is produced by running "Flag_for_Task==1"
			       String inputFile_AuthorName_AuthorID=inFile_6_AuthorName_AND_Auth_ID; 
		            //remove duplicates in name for p6 problem ( comment/uncomment as req)
			       readFile_eachLine_get_a_particular_token_calc_frequency.p6_remove_duplicates_of_authorName_in_freq_file(
		            												baseFolder,
							            							outputFileName, //authorName!!!Frequency  // INPUT
							            							inFile_6_AuthorName_AND_Auth_ID, //authName!!!authID // INPUT
							            							inFile_outputFileName_inFile_3_keyword_AND_freq_noduplicates // outFile_noDuplicates //OUTPUT
		            												);
		             
		           //note: ds10 database: some authors belongs to many domain->zeina karam
		            	
				System.out.println("out file:"+baseFolder+"outputUniqNames_auth_freq_no_duplicates.txt");
		    	
			    // ASSERT unique AUTHOR NAMES 
			    //consolidate ground truth to get Top N authors for each topic  ( comment/uncomment as req)
			    calc_ground_truth_2(
								  baseFolder,
								  output_Ground_Truth_removeDuplicate,  // inFile_output_Ground_Truth, (NO DUPLICATES)
								  inFile_6_AuthorName_AND_Auth_ID, 		// inFile_6_AuthorName_AND_Auth_ID
								  inFile_9_auth_id_doc_id_queryTopicRelated__applying_multi_crossAuthIDs, //inFile_9_auth_id_doc_id_queryTopicRelated__applying_multi_crossAuthIDs
								  inFile_10_authName_AND_auth_id__MAPPEDmultiAuthID_SORTED_add_minAuthID,
								  inFile_11_having_DIRTY_author_name,
								  inFile_12_having_NO_DIRTY_author_name,
								  inFile_outputFileName_inFile_3_keyword_AND_freq_noduplicates, //INPUT 
								  7, //{7} token_containing_mapString_each_Q_Topic_Freq,
								  baseFolder+"Ground_Truth_final_1_OUT.txt", //OUT
								  baseFolder+"Ground_Truth_final_2_OUT.txt", //OUT
								  baseFolder+"Ground_Truth_final_3_OUT.txt", //obsolete
								  baseFolder+"Ground_Truth_final_4_OUT.txt", //obsolete
								  baseFolder+"Ground_Truth_final_5_OUT.txt", //OUT
								  baseFolder+"Ground_Truth_final_6_OUT.txt", //OUT
								  baseFolder+"Ground_Truth_final_7_OUT.txt", //OUT
								  baseFolder+"Ground_Truth_final_8_OUT.txt", //OUT
								  baseFolder+"Ground_Truth_final_5_authName_freq_OUT.txt",
								  4000, //N_Top_authors
								  pattern_for_sourceChannel_TRAD,
								  pattern_for_sourceChannel_POLI,
								  false //isSOPprint
								  );
			    System.out.println("---flag_run_method==2 COMPLETED....in ground_truth.java");
		    }
		    
		    //NOTE: copy the output files from  "Ground_Truth_final_x_OUT.txt", "Ground_Truth_OUT_auth_id_CUMM_SUM_AVG_tf_idf.txt","Ground_Truth_final_5_authName_freq_OUT.txt" 
		    // "tf_idf_sum_avg.txt" from baseFolder to folder mentioned in var "base_groundTruth_folder"
		    
		    //STEP: compare just created above Top N file and Ground Truth 
		    String prefix="LDA"; // {LDA,PLSI}
		    
		    String suffix_run="_run1"; // example: _run1 or _run2
		    String base_groundTruth_folder=baseFolder+"ground_truth"+suffix_run+"/";
		    String base_LDA_folder=baseFolder+"LDA/"; //<--DONT CHANGE LDA to anything
		    String base_prefix_folder=base_LDA_folder+prefix+ suffix_run + "/";
		    //create these folders if it doesnt EXISTS
		    if(!new File(base_groundTruth_folder).exists()) new File(base_groundTruth_folder).mkdir();
		    if(!new File(base_LDA_folder).exists()) new File(base_LDA_folder).mkdir();
		    if(!new File(base_prefix_folder).exists()) new File(base_prefix_folder).mkdir();
		    //MANUALLY SET below 3

		    String top_N_groundTruth_file=base_groundTruth_folder+"Ground_Truth_final_"+GT_int+"_OUT.txt_Top_N.txt";
		    String top_N_groundTruth_file_TRAD=base_groundTruth_folder+"Ground_Truth_final_"+GT_int+"_OUT.txt_trad_.txt_Top_N.txt";
		    String top_N_groundTruth_file_POLI=base_groundTruth_folder+"Ground_Truth_final_"+GT_int+"_OUT.txt_poli_.txt_Top_N.txt";
		    
		    // note: use p6_ground_truth_plsi() method to convert a file having "d z probability" to a file below for each author.
		    //BELOW SAME AS outFile_each_auth_CUMM_topic_score, SO COMMENTED
//		    String lda_R_hlda_R_plsi_Top_N=base_prefix_folder+prefix+"_outFile_each_auth_CUMM_topic_score.txt";
		    String output_matched=base_prefix_folder+prefix+"_outFile_each_auth_CUMM_topic_score_matched_GT_.txt";
		    String output_matched_TRAD=base_prefix_folder+prefix+"_outFile_each_auth_CUMM_topic_score_matched_GT_TRAD.txt";
		    String output_matched_POLI=base_prefix_folder+prefix+"_outFile_each_auth_CUMM_topic_score_matched_GT_POLI.txt";
		    
		    String []arr_top_N_files=new String[30];
		    // copy below files to runX folder
		    arr_top_N_files[0]=baseFolder+"Ground_Truth_final_1_OUT.txt_Top_N.txt";
		    arr_top_N_files[1]=baseFolder+"Ground_Truth_final_1_OUT.txt_poli_.txt_Top_N.txt";
		    arr_top_N_files[2]=baseFolder+"Ground_Truth_final_1_OUT.txt_trad_.txt_Top_N.txt";
		    
		    arr_top_N_files[3]=baseFolder+"Ground_Truth_final_2_OUT.txt_Top_N.txt";
		    arr_top_N_files[4]=baseFolder+"Ground_Truth_final_2_OUT.txt_poli_.txt_Top_N.txt";
		    arr_top_N_files[5]=baseFolder+"Ground_Truth_final_2_OUT.txt_trad_.txt_Top_N.txt";
		    
		    arr_top_N_files[6]=baseFolder+"Ground_Truth_final_3_OUT.txt_Top_N.txt";
		    arr_top_N_files[7]=baseFolder+"Ground_Truth_final_3_OUT.txt_poli_.txt_Top_N.txt";
		    arr_top_N_files[8]=baseFolder+"Ground_Truth_final_3_OUT.txt_trad_.txt_Top_N.txt";
		    
		    arr_top_N_files[9] =baseFolder+"Ground_Truth_final_4_OUT.txt_Top_N.txt";
		    arr_top_N_files[10]=baseFolder+"Ground_Truth_final_4_OUT.txt_poli_.txt_Top_N.txt";
		    arr_top_N_files[11]=baseFolder+"Ground_Truth_final_4_OUT.txt_trad_.txt_Top_N.txt";
		    
		    arr_top_N_files[12]=baseFolder+"Ground_Truth_final_5_OUT.txt_Top_N.txt";
		    arr_top_N_files[13]=baseFolder+"Ground_Truth_final_5_OUT.txt_poli_.txt_Top_N.txt";
		    arr_top_N_files[14]=baseFolder+"Ground_Truth_final_5_OUT.txt_trad_.txt_Top_N.txt";
		    
		    arr_top_N_files[15]=baseFolder+"Ground_Truth_final_6_OUT.txt_Top_N.txt";
		    arr_top_N_files[16]=baseFolder+"Ground_Truth_final_6_OUT.txt_poli_.txt_Top_N.txt";
		    arr_top_N_files[17]=baseFolder+"Ground_Truth_final_6_OUT.txt_trad_.txt_Top_N.txt";
		    
		    arr_top_N_files[18]=baseFolder+"Ground_Truth_final_7_OUT.txt_Top_N.txt";
		    arr_top_N_files[19]=baseFolder+"Ground_Truth_final_7_OUT.txt_poli_.txt_Top_N.txt";
		    arr_top_N_files[20]=baseFolder+"Ground_Truth_final_7_OUT.txt_trad_.txt_Top_N.txt";
		    
		    arr_top_N_files[21]=baseFolder+"tf_idf_sum_avg.txt";
		    arr_top_N_files[22]=baseFolder+"outputUniqNames_auth_DistinctDocs_no_duplicates.txt";
		    arr_top_N_files[23]=baseFolder+"Ground_Truth_OUT_auth_id_CUMM_SUM_AVG_tf_idf.txt";
		    arr_top_N_files[24]=baseFolder+"Ground_Truth_OUT_auth_id_CUMM_SUM_AVG_tf_idf_noDup.txt";
		    
		    //lda files
		    arr_top_N_files[25]=baseFolder+"LDA_d_by_z_prob.txt";
		    arr_top_N_files[26]=baseFolder+"LDA_d_by_z_prob_TITLE.txt";
		    arr_top_N_files[27]=baseFolder+"LDA_d_by_z_prob_AVG.txt";
		     //
//		    arr_top_N_files[28]=baseFolder+"Ground_Truth_final_8_OUT.txt_Top_N.txt";
		    arr_top_N_files[28]=baseFolder+"Ground_Truth_final_8_OUT.txt_poli_.txt_Top_N.txt";
		    arr_top_N_files[29]=baseFolder+"Ground_Truth_final_8_OUT.txt_trad_.txt_Top_N.txt";
		    
		    //copy LDA run..
		    if(flag_run_method==3|| flag_run_method==99 || flag_run_method==34  || flag_run_method==234){
		    	
		    	//copy_Top_N_files_to_runX_folder ( comment and uncomment )
		    	copy_Top_N_files_to_runX_folder(
		    									arr_top_N_files,
		    									base_groundTruth_folder,
		    									base_prefix_folder // LDA/LDA_run1 folder
		    									);
		    	
		    	// processing LDA mallet output ( comment and uncomment ) 
		    	p6_ground_truth_plsi.
		        read_plsi_output_file_d_z_probability(
					                                        baseFolder,
					                                        inFile_1_output_file_plsi, // LDA or PLSI file
					                                        inFile_2_authorID_docIDCSV,
					                                        inFile_3_authorName_AuthID,
					                                        inFile_10_authName_AND_auth_id__MAPPEDmultiAuthID_SORTED_add_minAuthID, //convert to minAUTHID
					                                        outFile_each_auth_CUMM_topic_score, // OUTPUT
					                                        outFile_each_auth_CUMM_topic_score_TRAD, // OUTPUT
					                                        outFile_each_auth_CUMM_topic_score_POLI, // OUTPUT
					                                        latent_topic_manuallyFound_CSV,
					                                        pattern_for_sourceChannel_TRAD,
					                                        pattern_for_sourceChannel_POLI
					                                        );
		    	// TRAD  ( comment and uncomment )
		    	calc_ground_truth_compare_LDA_OR_hlda_plsi( baseFolder,
															top_N_groundTruth_file_TRAD,  //input (GROUND TRUTH)
															outFile_each_auth_CUMM_topic_score_TRAD, //input (OUTPUT FROM LDA OR PLSI OR HLDA algo) <-from p6_ground_truth_plsi.read_plsi_output_file_d_z_probability
															CSV_containing_10_interested_topics,
															top_N_match, // top_n_match
															output_matched_TRAD, // OUTPUT
															outputFile_missing_AUTHID_while_MATCHING_from_Flag_3,
															false //is_Append_4_outputFile_missing_AUTHID_while_MATCHING_from_Flag_3
		    												);
		    	// POLI  ( comment and uncomment )
		    	calc_ground_truth_compare_LDA_OR_hlda_plsi( baseFolder,
															top_N_groundTruth_file_POLI,  //input (GROUND TRUTH)
															outFile_each_auth_CUMM_topic_score_POLI, //input (OUTPUT FROM LDA OR PLSI OR HLDA algo) <-from p6_ground_truth_plsi.read_plsi_output_file_d_z_probability
															CSV_containing_10_interested_topics,
															top_N_match, //top_n_match
															output_matched_POLI, // OUTPUT
															outputFile_missing_AUTHID_while_MATCHING_from_Flag_3,
															true //is_Append_4_outputFile_missing_AUTHID_while_MATCHING_from_Flag_3
		    												);
		    	
			    //OBSOLETE(BELOW RUNS for TRAN and POLI together ranking without individual ranking for each ) 
		    	//compare ground truth and Top N from hlda or lda or plsi prediction 
//			    calc_ground_truth_compare_LDA_OR_hlda_plsi( baseFolder,
//			    											top_N_groundTruth_file,  //input (GROUND TRUTH)
//			    											outFile_each_auth_CUMM_topic_score, //input (OUTPUT FROM LDA OR PLSI OR HLDA algo) <-from p6_ground_truth_plsi.read_plsi_output_file_d_z_probability
//			    											CSV_containing_10_interested_topics,
//			    											top_N_match, //top_n_match
//			    											output_matched // OUTPUT
//			    										  );
			    
			    System.out.println("source : base_groundTruth_folder:"+base_groundTruth_folder);
			    System.out.println("output_matched:"+output_matched);
			    System.out.println("base_prefix_folder:"+base_prefix_folder);
//			    System.out.println("lda_R_hlda_R_plsi_Top_N:"+lda_R_hlda_R_plsi_Top_N);
			    System.out.println("base_LDA_folder:"+base_LDA_folder);
			    System.out.println("output_matched_TRAD(out):"+output_matched_TRAD);
			    System.out.println("output_matched_POLI(out):"+output_matched_POLI);
			    System.out.println("LDA output with CUMM topic score:"+outFile_each_auth_CUMM_topic_score);
			    System.out.println("outFile_each_auth_CUMM_topic_score:"+outFile_each_auth_CUMM_topic_score);
//			    System.out.println("lda_R_hlda_R_plsi_Top_N:"+lda_R_hlda_R_plsi_Top_N);
			    System.out.println("----BEGIN-------output from read_plsi_output_file_d_z_probability");
			    System.out.println("\ninFile_1_output_file_plsi:"+inFile_1_output_file_plsi+
					                "\ninFile_2_authorID_docIDCSV:"+inFile_2_authorID_docIDCSV+
					                "\ninFile_3_authorName_AuthID:"+inFile_3_authorName_AuthID);
			    System.out.println("----END-------output from read_plsi_output_file_d_z_probability");
			    System.out.println("---flag_run_method==3 COMPLETED....in ground_truth.java.");
			    System.out.println("-------------------------------------------------------------");
		    }
		    
		    //MEAN AVERAGE PRECISION
		    int Top_N_max_for_calc_MAP=top_N_match;
		    String debugFileMAP=baseFolder+"debug_MAP.txt";
		    TreeMap<String, Double>  map_precision_map_POLI=new TreeMap<String, Double>();
		    // 
		    if(flag_run_method==4 || flag_run_method==99 || flag_run_method==34 || flag_run_method==234){
		    	 // TRAD
		    	 TreeMap<String, Double>  map_precision_map_TRAD=calc_mean_avg_precision_MAP(
																					baseFolder,
																					output_matched_TRAD, //input file
																					Top_N_max_for_calc_MAP, //=top_N_match
																					output_Result, // OUTPUT
																					false, // is_Append_output_results,
																					false, // is_Append_for_debugFile
																					false, // isSOPprint
																					Flag_choose_METHODtype_DCG,
																					is_convert_using_bucket_for_rankingScore,
																					is_use_original_ranking,
																					debugFileMAP
													    							);
		    	 System.out.println("*input 1:"+output_matched_TRAD);
		    	 System.out.println("*input 2:"+Top_N_max_for_calc_MAP);
		    	 // POLI
		    	 					map_precision_map_POLI=calc_mean_avg_precision_MAP(
																					baseFolder,
																					output_matched_POLI, //input file
																					Top_N_max_for_calc_MAP, //=top_N_match
																					output_Result, // OUTPUT
																					true,  // is_Append_output_results,
																					true,  // is_Append_for_debugFile
																					false, // isSOPprint
		    	 																	Flag_choose_METHODtype_DCG,
		    	 																	is_convert_using_bucket_for_rankingScore,
		    	 																	is_use_original_ranking,
																					debugFileMAP
										    										);
		    	 System.out.println("*input 1:"+output_matched_POLI);
			    //calc mean avg precision
		    	//OBSOLETE(BELOW RUNS for TRAN and POLI together ranking without individual ranking for each )
//			    calc_mean_avg_precision_MAP(
//			    								baseFolder,
//			    								output_matched, //input file
//			    								Top_N_max_for_calc_MAP,
//			    								output_Result,
//			    								false, //isSOPprint
//			    								debugFile
//			    							);
		    	FileWriter writerResults=new FileWriter(new File(output_Result), true);
		    	System.out.println("-------------------------------top_N_match:"+top_N_match+" GT_int:"+GT_int);
		    	System.out.println("TRAD precision:"+map_precision_map_TRAD.get("precision")+" MAP:"+map_precision_map_TRAD.get("map") 
		    								+" DCG:"+map_precision_map_TRAD.get("dcg") );
		    	writerResults.append("\n\nTRAD precision:"+map_precision_map_TRAD.get("precision")+" MAP:"+map_precision_map_TRAD.get("map")
		    								+" DCG:"+map_precision_map_TRAD.get("dcg"));
		    	if(map_precision_map_POLI!=null){
			    	if(map_precision_map_POLI.size()>0){
				    	System.out.println("POLI precision:"+map_precision_map_POLI.get("precision")+" MAP:"+map_precision_map_POLI.get("map")
				    									+" DCG:"+map_precision_map_POLI.get("dcg") );
				    	writerResults.append("\nPOLI precision:"+map_precision_map_POLI.get("precision")+" MAP:"+map_precision_map_POLI.get("map")
				    												+" DCG:"+map_precision_map_POLI.get("dcg"));
				    	System.out.println("precision(AVG above):"+ (map_precision_map_TRAD.get("precision")+map_precision_map_POLI.get("precision"))/2.0
		    					+" MAP(AVG above):"+ (map_precision_map_TRAD.get("map")+map_precision_map_POLI.get("map"))/2.0 );
				    	writerResults.append("\nprecision(AVG above):"+ (map_precision_map_TRAD.get("precision")+map_precision_map_POLI.get("precision"))/2.0
    							+" MAP(AVG above):"+ (map_precision_map_TRAD.get("map")+map_precision_map_POLI.get("map"))/2.0 );
				    	
				    	
				    	System.out.println("CSV below->\n\""+map_precision_map_TRAD.get("precision")+","+map_precision_map_TRAD.get("map")+","+map_precision_map_TRAD.get("dcg")
				    						+","+map_precision_map_POLI.get("precision")+","+map_precision_map_POLI.get("map")+","+map_precision_map_POLI.get("dcg")
				    						+"#\"+"
				    						);
				    	System.out.println("Copy these CSV lines for postResult_PIVOT_table()");
			    	}
		    	}

		    	
		    	writerResults.flush(); 
		    	System.out.println("-------------------------------");
		    	System.out.println("output_matched_TRAD(input):"+output_matched_TRAD);
		    	System.out.println("output_matched_POLI(input):"+output_matched_POLI);
			    System.out.println("output_matched:"+output_matched);
			    System.out.println("GIVEN pattern_for_sourceChannel_TRAD:"+pattern_for_sourceChannel_TRAD+" pattern_for_sourceChannel_POLI:"+pattern_for_sourceChannel_POLI);
			    System.out.println("---flag_run_method==4 COMPLETED.....in ground_truth.java");
		    }
		    
		    
		    System.out.println("--------------------------------------------");
		    System.out.println("LDA output with CUMM topic score:"+outFile_each_auth_CUMM_topic_score);
		     // for each q_topic and each author , find the count found in documents
		    //OBSELETE This method was created as part of count not working in "calc_ground_truth", NOT TESTED
//		    calc_ground_truth_1_2(  inFile_3_with_all_tokens,
//								    token_no_contain_authors_in_inFile_3,
//								    token_no_contain_bodyText_in_inFile_3,
//		    					  	CSV_containing_10_interested_topics,
//		    					  	outputFileName_inFile_3_keyword_AND_freq
//		    					 );
		   
		    
			System.out.println("Time Taken:" + NANOSECONDS.toSeconds(System.nanoTime() - t0) +" seconds; "
 	   				  			+ (NANOSECONDS.toSeconds(System.nanoTime() - t0))/60 +" minutes");

	}
}