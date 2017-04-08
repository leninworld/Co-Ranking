

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.TreeMap;
import java.util.Map.Entry;

import crawler.readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile;

public class create_by_converting_tfidf_FILE_2_SVMrankINPUTfile {
	
	// convert_tfIDF_files_to_inputForSVMrank
	public static void convert_tfIDF_files_to_inputForSVMrank(
														String inputFolder_having_SETofINPUTfiles,
														String outputFile_trainFILE,
														String outputFile_testFILE
														){
		try {
			String [] arr_files=new File(inputFolder_having_SETofINPUTfiles).list();

			if(new File(outputFile_trainFILE).exists())
				new File(outputFile_trainFILE).delete();
			
			if(new File(outputFile_testFILE).exists())
				new File(outputFile_testFILE).delete();
			
			//each file is a new q_id
			FileWriter writer_train=new FileWriter(outputFile_trainFILE);
			FileWriter writer_test=new FileWriter(outputFile_testFILE);
			
			
			int q_id=0;
			TreeMap<Integer,String> map_inFile_EachLine_of_document=new TreeMap<Integer, String>();
			int c=0;
			//
			while(c<arr_files.length){
				if(arr_files[c].toLowerCase().indexOf("store")>=0){c++;continue;}
				System.out.println("Currently running file:"+inputFolder_having_SETofINPUTfiles+arr_files[c]);
				String curr_inFile=inputFolder_having_SETofINPUTfiles+arr_files[c];
				
				q_id++;
				
				// loading
				map_inFile_EachLine_of_document=readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
						 						.readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line( curr_inFile, 
						 																									 -1, //startline, 
																															 -1, //endline,
																															 " map_inFile_EachLine_of_document ", //debug_label
																															 false //isPrintSOP
																															 );

		    					
//				3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A
//				2 qid:1 1:0 2:0 3:1 4:0.1 5:1 # 1B 
//				1 qid:1 1:0 2:1 3:0 4:0.4 5:0 # 1C
//				1 qid:1 1:0 2:0 3:1 4:0.3 5:0 # 1D  
				
				//
				for(int seq:map_inFile_EachLine_of_document.keySet()){
					String curr_line=map_inFile_EachLine_of_document.get(seq);
					String [] arr_=curr_line.split("!!!");
					
					String curr_docID=arr_[0];
					String curr_FEATURES_n_tfIdf=arr_[1];
					
					if(curr_line.length()>1){
						//
						writer_train.append(curr_docID+"\t"+ "qid:"+q_id+"\t"+curr_FEATURES_n_tfIdf+"\n" );
						writer_train.flush();
					}
					
				}
				
				c++;
		
			}
		 
			
		} catch (Exception e) {
			// TODO: handle exception
		}
	}
	
	
	// main
	public static void main(String[] args) throws IOException {
		 
	//note: FILES IN this folder is created by METHOD "convert_DocID_WordID_Freq_To_another_format_2"
	String  inputFolder_having_SETofINPUTfiles="/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/tempFolder4Conversion/TF-IDF/";
	int 	GT_num=8;
	String  baseFolder="/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/";
	String  input_authID_docID=baseFolder+"auth_id_doc_id.txt_NO_dirtyAuthNames.txt";
	String  input_authID_docID2=baseFolder+	"auth_id_doc_id_queryTopicRelated.txt_NO_dirtyAuthNames.txt";
	
	// OUTPUT 
	String outFileTRAIN=inputFolder_having_SETofINPUTfiles+"inputSVMrank_train.txt";
	String outFileTEST=inputFolder_having_SETofINPUTfiles+"inputSVMrank_test.txt";
	
	String groundTruthFolder="ground_truth_run1";
	String ground_truthFile_TRAD=baseFolder+groundTruthFolder+"/Ground_Truth_final_"+GT_num+"_OUT.txt_trad_.txt_Top_N.txt";
	String ground_truthFile_POLI=baseFolder+groundTruthFolder+"/Ground_Truth_final_"+GT_num+"_OUT.txt_poli_.txt_Top_N.txt";
	
	String queries_CSV="boko haram,crime,election,india,syria,climate change";
	
	String [] arr_query_atom=queries_CSV.split(",");
	
	TreeMap<String,TreeMap<Integer, String>>  map_topic_authID_docIDcsv=new TreeMap<String, TreeMap<Integer, String>>();
	TreeMap<Integer, String>  map_authID_docIDcsv_temp=new TreeMap<Integer, String>();
	TreeMap<Integer, String>  map_authID_docIDcsv=new TreeMap<Integer, String>();
	TreeMap<Integer, String>  map_authID_docIDcsv_temp2=new TreeMap<Integer, String>();
	TreeMap<Integer, String>  map_authID_docIDcsv_temp3=new TreeMap<Integer, String>();
	
	String [] arr_files=new File(inputFolder_having_SETofINPUTfiles).list();
	
	
	// each line has <!!!authID!!!docIDinCSV>
	  TreeMap<Integer, String>  map_seq_inFile_TEMP2=
			  readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
								.readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
																								  input_authID_docID2, 
																							 	 -1, //startline, 
																								 -1, //endline,
																								 " loading", //debug_label
																								 false //isPrintSOP
																								 );
		  //ex :!!!29!!!1634,1637,1638,1744,1754 
		  for(int seq:map_seq_inFile_TEMP2.keySet()){
			  String [] arr_ = map_seq_inFile_TEMP2.get(seq).split("!!!");
			  map_authID_docIDcsv_temp2.put(Integer.valueOf(arr_[0]), arr_[1]);
		  }
	
	
	 int c6=0;
	 	// 
		 while(c6<arr_query_atom.length){
			 	  map_authID_docIDcsv_temp=new TreeMap<Integer, String>();
			 	   String curr_topic_authID_docID=baseFolder+arr_query_atom[c6]+"_authID_docIDCSV__.txt";
			 	   
				  // topic-wise have authID and docIDinCSV
				  // STEP 1 : Loading seq and Line 
				 // each line has <!!!authID!!!docIDinCSV>
				  TreeMap<Integer, String>  map_seq_inFile_TEMP=
						  readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
											.readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
																											 curr_topic_authID_docID, 
																										 	 -1, //startline, 
																											 -1, //endline,
																											 " loading", //debug_label
																											 false //isPrintSOP
																											 );
					  //ex :!!!29!!!1634,1637,1638,1744,1754 
					  for(int seq:map_seq_inFile_TEMP.keySet()){
						  String [] arr_ = map_seq_inFile_TEMP.get(seq).split("!!!");
						  map_authID_docIDcsv_temp.put(Integer.valueOf(arr_[1]), arr_[2]);
					  }
					  map_topic_authID_docIDcsv.put(arr_query_atom[c6], map_authID_docIDcsv_temp);
					  c6++;
		 } // while(c6<arr_query_atom.length){
	 
			//thit has for each AuthID, consolidated wordID (featureID) from all documents corresponding to the author.
		  TreeMap<String, TreeMap<Integer, TreeMap<Integer, Double>>> map_topicNAME_authID_featureID_N_value_GLOBAL=
				  			new TreeMap<String, TreeMap<Integer, TreeMap<Integer, Double>>>();
		  
			int c=0;

			//thit has for each AuthID, consolidated wordID (featureID) from all documents corresponding to the author.
			TreeMap<Integer, TreeMap<Integer, Double>> map_authID_featureID_N_value=new TreeMap<Integer, TreeMap<Integer, Double>>();
			
			TreeMap<String,TreeMap<Integer, String>> map_topic_docID_featureNvaluesCSV_4_currTopic=new TreeMap<String,TreeMap<Integer, String>>();
			
			TreeMap<Integer, String> map_docID_featureNvaluesCSV_4_currTopic=new TreeMap<Integer, String>();
			
			// STEP 2 ----- 
			// iterate each topic-wise file (topic: syria, india etc)
			while(c<arr_files.length){
				
				 
				if(arr_files[c].toLowerCase().indexOf("store")>=0){c++;continue;}
				System.out.println("Currently running file:"+inputFolder_having_SETofINPUTfiles+arr_files[c]);
				String curr_inFile_4_currTOPIC=inputFolder_having_SETofINPUTfiles+arr_files[c];
				
//				  // extract topic name from filename
				  String curr_topicName_extracted=curr_inFile_4_currTOPIC.replace( "tf-idf-for-", "").replace("_OUT.txt_PAGERANK_TFIDF_OUT.txt", "")
						  				.replace("/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/tempFolder4Conversion/TF-IDF/", "");
				
				  if(new File(curr_inFile_4_currTOPIC).isDirectory()){
					  c++; continue;
				  }
				  
				  //EACH LINE has <docID!!!feature1:tf-idfvalue feature2:tf-idfvalue>
				  TreeMap<Integer, String>  map_seq_inFile_4_currTOPIC=
						  readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
											.readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
																											 curr_inFile_4_currTOPIC, 
																										 	 -1, //startline, 
																											 -1, //endline,
																											 " loading", //debug_label
																											 false //isPrintSOP
																											 );
				  
					   // (from second file) iterate each docID and get featureID and valueID
					  for(int seq:map_seq_inFile_4_currTOPIC.keySet()){
						  String [] arr_docID_featureIDnVALUES=map_seq_inFile_4_currTOPIC.get(seq).split("!!!");
						  map_docID_featureNvaluesCSV_4_currTopic.put(Integer.valueOf(arr_docID_featureIDnVALUES[0]), 
								  									  arr_docID_featureIDnVALUES[1]
								  									  );
					  }
					  //
					  map_topic_docID_featureNvaluesCSV_4_currTopic.put(curr_topicName_extracted , map_docID_featureNvaluesCSV_4_currTopic);
					  map_docID_featureNvaluesCSV_4_currTopic=new TreeMap<Integer, String>();
					  c++;
			}
					  TreeMap<Integer, Double> map_featureID_N_value_4_currAuthID_CONSOLIDATE_temp =new TreeMap<Integer, Double>();
//					  
//					   
//					  // extract topic name from filename
//					  String curr_topicName_extracted=curr_inFile_4_currTOPIC.replace( "tf-idf-for-", "").replace("_OUT.txt_PAGERANK_TFIDF_OUT.txt", "")
//							  				.replace("/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/tempFolder4Conversion/TF-IDF/", "");
//					  
					  // iterate each topic, get all corresponding authors   
					  for(String currTopic:map_topic_authID_docIDcsv.keySet()){
						  int passed_docIDs=0;int NOT_passed_docIDs=0;
						  
						  map_authID_docIDcsv=map_topic_authID_docIDcsv.get(currTopic);
						  
						  map_authID_docIDcsv=map_authID_docIDcsv_temp2; //overwrite
						  
						  map_authID_docIDcsv_temp2=new TreeMap<Integer, String>();
						  for(int seq:map_seq_inFile_TEMP2.keySet()){
							  String [] arr_ = map_seq_inFile_TEMP2.get(seq).split("!!!");
							  if(arr_[2].indexOf(currTopic)>=0) //match on curr topic
								  map_authID_docIDcsv_temp2.put(Integer.valueOf(arr_[0]), arr_[1]);
						  }
						  map_authID_docIDcsv=map_authID_docIDcsv_temp2;
						  
						  System.out.println("curr topic:"+currTopic+" authID_docIDcsv:"+map_authID_docIDcsv.size());
						  
						  map_docID_featureNvaluesCSV_4_currTopic=map_topic_docID_featureNvaluesCSV_4_currTopic.get(currTopic);
						  
						  //iterate each author, get all corresponding docid's and get consolidated tf-idf (from first file)
						  for(int curr_authID:map_authID_docIDcsv.keySet()){
   
							  String [] arr_docIDs_of_currAuthID_ATOM=map_authID_docIDcsv.get(curr_authID).split(",");
							  int c2=0;
							  
//							  if(map_authID_featureID_N_value.containsKey(curr_authID)){
//								  map_featureID_N_value_4_currAuthID_CONSOLIDATE_temp = map_authID_featureID_N_value.get(curr_authID);
//							  }
//							  else{
//
//							  }
							  // each DOCUMENT of current AuthID
							  while(c2<arr_docIDs_of_currAuthID_ATOM.length){
								  
								  int curr_docID_atom=Integer.valueOf(arr_docIDs_of_currAuthID_ATOM[c2]);
								  String curr_featureNvaluesCSV_4_currTopic_AND_currDocID=map_docID_featureNvaluesCSV_4_currTopic.get(curr_docID_atom);
								  
								   if(curr_featureNvaluesCSV_4_currTopic_AND_currDocID==null ) {
//									   System.out.println("curr_docID_atom-->"+curr_docID_atom+ "<---NULL"+" currTopic:"+currTopic);
									   NOT_passed_docIDs++;
									   c2++;continue;
								   }
								   else{
									   passed_docIDs++;
								   }
								  
	//							  System.out.println("curr_featureNvaluesCSV_4_currTopic_AND_currDocID:"+curr_featureNvaluesCSV_4_currTopic_AND_currDocID
	//									  				+" map_docID_featureNvaluesCSV_4_currTopic:"+map_docID_featureNvaluesCSV_4_currTopic);
								  String [] arr_featureNvalue_atom=curr_featureNvaluesCSV_4_currTopic_AND_currDocID.split(" ");
								  
								  int c3=0;
								  while(c3<arr_featureNvalue_atom.length){
									  String [] arr_feature_N_value_PAIR=arr_featureNvalue_atom[c3].split(":");
									  
									  int 	 featureID=Integer.valueOf(arr_feature_N_value_PAIR[0]);
									  double featureValue=Double.valueOf(arr_feature_N_value_PAIR[1]);
									  // 
									  if(!map_featureID_N_value_4_currAuthID_CONSOLIDATE_temp.containsKey(featureID)){
										  map_featureID_N_value_4_currAuthID_CONSOLIDATE_temp.put(featureID, featureValue);
									  }
									  else{
										  double avgFeatureValue=(map_featureID_N_value_4_currAuthID_CONSOLIDATE_temp.get(featureID)
												  					+featureValue) / (double)2;
										  map_featureID_N_value_4_currAuthID_CONSOLIDATE_temp.put(featureID, avgFeatureValue);
									  }
									  c3++;
								  } // while(c3<arr_featureNvalue_atom.length){
								  c2++;
							  } //while(c2<arr_docIDs_of_currAuthID_ATOM.length){
							  
							  // each document of current author
							  // this has for each AuthID, consolidated wordID (featureID) from all documents corresponding to the author.
							  // UPDATE consolidated at this iteration
							  if(map_featureID_N_value_4_currAuthID_CONSOLIDATE_temp.size()>0)
								  map_authID_featureID_N_value.put( curr_authID , 
										  							map_featureID_N_value_4_currAuthID_CONSOLIDATE_temp);
							  
//							  System.out.println("curr topic:"+currTopic+" authID_docIDcsv:"+map_authID_docIDcsv.size()+" passed_docIDs:"+passed_docIDs
//									  		    +" NOT_passed_docIDs:"+NOT_passed_docIDs);

					  	} // for(int curr_authID:map_authID_docIDcsv.keySet()){
						  map_topicNAME_authID_featureID_N_value_GLOBAL.put(currTopic, map_authID_featureID_N_value);						  
//						  map_featureID_N_value_4_currAuthID_CONSOLIDATE_temp=new TreeMap<Integer, Double>();
						  // save topic-wise , 
						  
					  } // for(String currTopic:map_topic_authID_docIDcsv.keySet()){
					  
//					  System.out.println("map_authID_featureID_N_value.size:"+map_authID_featureID_N_value.size()
//							  			+" for topic-wise file:"+currTopic);
			

//					  map_authID_featureID_N_value=new TreeMap<Integer, TreeMap<Integer, Double>>();
					  c++;
//			} //while(c<arr_files.length){
			
			TreeMap<String, String> map_currLine_currLine=new TreeMap<String, String>();
			
			map_currLine_currLine=
					readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
					readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile( 
																								  ground_truthFile_TRAD
																								, -1
																								, -1
																								, "" //outfile
																								, false // is_Append_outFile
																								, false
																								, "debug_label"
																								, -1 // token for  Primary key
																								, false//isSOPprint
																								);
			
			int 	lineNo=0;
			int    curr_authID=-1; String curr_authName=""; String curr_q_label_AND_rank="";
			
			TreeMap<String, Integer> map_curr_q_topic_lines_in_GroundTruth_authName_rankGT=new TreeMap<String, Integer>();
			TreeMap<Integer, Integer> map_curr_q_topic_lines_in_GroundTruth_authID_rankGT=new TreeMap<Integer, Integer>();
			
			TreeMap<String,TreeMap<Integer, Integer>> map_topic_authID_rankGT_4_GroundTruth=new TreeMap<String,TreeMap<Integer, Integer>>();
			TreeMap<String,TreeMap<String, Integer>> map_topic_authName_rankGT_4_GroundTruth=new TreeMap<String,TreeMap<String, Integer>>();
			
			int c5=0;
			// STEP 3 ----- get GROUND TRUTH RANKING FOR topic-wise authors 
			// iterate each query
			while(c5<arr_query_atom.length){
				
				String curr_query=arr_query_atom[c5];
				 map_curr_q_topic_lines_in_GroundTruth_authName_rankGT=new TreeMap<String, Integer>();
				// curr_ground_truth_line (ONLY matching curr_q_topic)
				for(String curr_ground_truth_line:map_currLine_currLine.keySet()){
					lineNo++;
	//				if(lineNo<=10)
						
					if(curr_ground_truth_line.indexOf("!!!")==-1) {
						System.out.println("SKIPPING curr_ground_truth_line:"+curr_ground_truth_line);
						continue;} //skip header
					// 
					if(curr_ground_truth_line.indexOf(curr_query+"(") ==-1)
						continue;
					
					//
					if(
	//					curr_ground_truth_line.indexOf("!!!"+curr_q_topic)>=0 && 
					    curr_ground_truth_line.indexOf("!!!")>=0 ){
	//					System.out.println("INSIEE ------");
						String [] s=curr_ground_truth_line.split("!!!");
						curr_authName=s[2]; //authorName			
						curr_authID= Integer.valueOf(s[0].replace("auth_id:", ""));
						
						curr_authName=p6_MAIN_Work.p6_normalize_auth_name(curr_authName).replace("auth:from: ", "")
											 .replace("auth:from:",""); 
						
						curr_q_label_AND_rank=curr_ground_truth_line.split("!!!")[3]; //contains "curr_q_label(rank)" -> syria(1)
						
						String curr_q_label_from_GT=
									curr_q_label_AND_rank.substring(0 , curr_q_label_AND_rank.indexOf("("));
						
						curr_q_label_from_GT=p6_MAIN_Work.p6_normalize_auth_name(curr_q_label_from_GT).replace("auth:from: ", "")
												    .replace("auth:from:","");
						
						int rank_from_GT=new Integer(
											curr_q_label_AND_rank.substring( curr_q_label_AND_rank.indexOf("("),
																		     curr_q_label_AND_rank.length()).replace("(", "")
																		    .replace(")", ""));
						
	//					System.out.println("^^^^^^^^^^label(GT):"+curr_q_label_from_GT+" rank(GT):"+rank_from_GT);
						map_curr_q_topic_lines_in_GroundTruth_authName_rankGT.put(curr_authName, rank_from_GT);
						map_curr_q_topic_lines_in_GroundTruth_authID_rankGT.put(curr_authID, rank_from_GT);
					}
					else{ //debug
	//					System.out.println("NOT:"+curr_ground_truth_line+"<-->"+curr_ground_truth_line.indexOf(curr_q_topic)
	//										+"<--->"+curr_ground_truth_line.indexOf("!!!"));
					}
					
				} // END for(String curr_ground_truth_line:map_currLine_currLine.keySet()){
			
				// save ranking for all queries (Topics)
				map_topic_authName_rankGT_4_GroundTruth.put(curr_query, map_curr_q_topic_lines_in_GroundTruth_authName_rankGT);
				
				map_topic_authID_rankGT_4_GroundTruth.put(curr_query, map_curr_q_topic_lines_in_GroundTruth_authID_rankGT);
				System.out.println("curr_query:"+curr_query+ " size:"+map_curr_q_topic_lines_in_GroundTruth_authName_rankGT.size()  +
								" map_curr_q_topic_lines_in_GroundTruth_authName_rankGT:"+map_curr_q_topic_lines_in_GroundTruth_authName_rankGT);
				c5++;
			}
		  
			System.out.println("map_topicNAME_authID_featureID_N_value_GLOBAL.size:"+map_topicNAME_authID_featureID_N_value_GLOBAL.size());
			
			
			TreeMap<Integer,Integer>  map_uniqueAuthID_asKEY=new TreeMap<Integer, Integer>();
			int total_authID_for_allTopics=0;
			TreeMap<Integer,String>  map_AuthID_concatenateFeatureIDfeatureValue_pastSaved=new TreeMap<Integer, String>();
			int q_id=0;
			// topic-wise 
			for(String topic:map_topicNAME_authID_featureID_N_value_GLOBAL.keySet()){
				 System.out.println("topic:"+topic+" authorsMap size:"+map_topicNAME_authID_featureID_N_value_GLOBAL.get(topic).size() );
				 FileWriter writer=new FileWriter(new File(baseFolder+topic+"_out.txt"));
				 TreeMap<Integer, TreeMap<Integer, Double>> map_authID_featIDfeatValue=map_topicNAME_authID_featureID_N_value_GLOBAL.get(topic);
				 
				 //ranking
				 TreeMap<Integer,Integer>  map_authID_rank_4_currTopic=map_topic_authID_rankGT_4_GroundTruth.get(topic);
				  
				 String concCurrLine="";
				 q_id++;
				 int last_autoAssignedRank=1000;
				 //each author
				 for(int authID:map_authID_featIDfeatValue.keySet()){
					 map_uniqueAuthID_asKEY.put(authID , -1);
					 
					 // 
					 if(map_AuthID_concatenateFeatureIDfeatureValue_pastSaved.containsKey(authID)){
						 concCurrLine=map_AuthID_concatenateFeatureIDfeatureValue_pastSaved.get(authID);
					 }
					 else{
						 TreeMap<Integer,Double> map_currFeatureID_featureValue=map_authID_featIDfeatValue.get(authID);
						 //each feature
						 for(int curr_FeatureID:map_currFeatureID_featureValue.keySet()){
							 //
							 if(concCurrLine.length()==0){
								 concCurrLine=curr_FeatureID+":"+map_currFeatureID_featureValue.get(curr_FeatureID);
							 }
							 else{
								 concCurrLine+=" "+curr_FeatureID+":"+map_currFeatureID_featureValue.get(curr_FeatureID);
							 }
						 }
					 }
					 int rank_for_currAuthor_4currTopic=-1;
					 //
					 if(map_authID_rank_4_currTopic.containsKey(curr_authID))
						 rank_for_currAuthor_4currTopic=map_authID_rank_4_currTopic.get(curr_authID);
					 else{
						 last_autoAssignedRank++;
					 }
					 
					 if(rank_for_currAuthor_4currTopic==-1)
						 rank_for_currAuthor_4currTopic=last_autoAssignedRank;
						 
					 // writing
					 writer.append(rank_for_currAuthor_4currTopic+" qid:"+q_id+ " "+concCurrLine+" #"+authID+"\n");
					 writer.flush();
					 
					 if(!map_AuthID_concatenateFeatureIDfeatureValue_pastSaved.containsKey(authID))
						 map_AuthID_concatenateFeatureIDfeatureValue_pastSaved.put(authID, concCurrLine);
					 
					 concCurrLine="";
				 }
				 
				total_authID_for_allTopics+=map_topicNAME_authID_featureID_N_value_GLOBAL.get(topic).size() ;
			}
			
			System.out.println("total_authID_for_allTopics:"+total_authID_for_allTopics
							  +" map_uniqueAuthID_asKEY.size:"+map_uniqueAuthID_asKEY.size());
			
			int total_loaded_authID_of_allTopics_inclDup=0;
			for( String topic:map_topic_authID_docIDcsv.keySet()){
				System.out.println("topic:"+topic+" loaded lines(authID_docIDcsv):"+map_topic_authID_docIDcsv.get(topic).size());
				total_loaded_authID_of_allTopics_inclDup+=map_topic_authID_docIDcsv.get(topic).size();
			}
			System.out.println("total_loaded_authID_of_allTopics_inclDup:"+total_loaded_authID_of_allTopics_inclDup);
			System.out.println("map_authID_docIDcsv_temp2.size:"+map_authID_docIDcsv_temp2.size());
			System.out.println("map_topic_authName_rankGT_4_GroundTruth:"+map_topic_authName_rankGT_4_GroundTruth);
			
//			System.out.println(""+map_topicNAME_authID_featureID_N_value_GLOBAL.get("climate change"));
			
			// topic-wise - split to two 
			// create 2-fold cross validation files (2 files)
			
			
		// convert_tfIDF_files_to_inputForSVMrank
//		convert_tfIDF_files_to_inputForSVMrank( inputFolder_having_SETofINPUTfiles,
//												outFileTRAIN,
//												outFileTEST
//												);
	
	}
	
}
