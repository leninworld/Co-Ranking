

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.TreeMap;

import crawler.readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile;

 
public class p6_ground_truth_plsi {
   
    // read_plsi_output_file_d_z_probability
    public static TreeMap<Integer, Integer> read_plsi_output_file_d_z_probability(
                                                                                String baseFolder,
                                                                                String inFile_1_output_file_plsi,
                                                                                String inFile_2_authorID_docIDCSV,
                                                                                String inFile_3_authorName_AuthID,
                                                                                String inFile_10_authName_AND_auth_id__MAPPEDmultiAuthID_SORTED_add_minAuthID,
                                                                                String outFile_each_auth_CUMM_topic_score, //OUT
                                                                                String outFile_each_auth_CUMM_topic_score_TRAD, //OUT
                                                                                String outFile_each_auth_CUMM_topic_score_POLI, //OUT
                                                                                String latent_topic_manuallyFound_CSV,
                                                                                String pattern_for_sourceChannel_TRAD,
                                                                                String pattern_for_sourceChannel_POLI 
                                                                                ){
    	
    
          
        BufferedReader reader=null;
        String line=""; FileWriter writer=null;
        FileWriter writer_TRAD=null;FileWriter writer_POLI=null;
        TreeMap<Integer, Integer> mapAuthId_unique_old_new=new TreeMap<Integer, Integer>();
        TreeMap<Integer, String> mapDistinct_authID_authName=new TreeMap<Integer, String>();
        
        TreeMap<Integer, String> mapDistinct_authID_sourceChannel=new TreeMap<Integer, String>();
        TreeMap<Integer,String>map_eachLine_inFile10__MAPPEDmultiAuthID_SORTED_add_minAuthID=new TreeMap<Integer, String>();
        
        String[] z_q_topic_manually_found=latent_topic_manuallyFound_CSV.split(",");
        TreeMap<Integer, Integer> mapDistinct_topicID_as_KEY=new TreeMap<Integer,Integer>();
        int max_=-1;
        try{
        	System.out.println("input file:"+inFile_10_authName_AND_auth_id__MAPPEDmultiAuthID_SORTED_add_minAuthID);
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
        	// ******* below for loop COPIED from ground_truth.java
			//mapping min_AUTHID and ALL_authIDs
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
			
			
        	int j=0;
        	while(j<z_q_topic_manually_found.length){
        		mapDistinct_topicID_as_KEY.put(j, -1);
        		j++;
        	}
        	
            TreeMap<Integer, Integer> mapDistinct_authID_as_KEY=new TreeMap<Integer,Integer>();// this loaded from inFile_2_authorID_docIDCSV
            
            FileWriter writerDebug=new FileWriter(new File(baseFolder+"debug_PLSI_ground_truth.txt"));
            writer =new FileWriter(new File(outFile_each_auth_CUMM_topic_score));
            writer_TRAD =new FileWriter(new File(outFile_each_auth_CUMM_topic_score_TRAD));
            writer_POLI =new FileWriter(new File(outFile_each_auth_CUMM_topic_score_POLI));
            
            
            reader=new BufferedReader(new FileReader(new File(inFile_3_authorName_AuthID) ));
            // each Line (authName!!!authID!!!sourceChannel)
            while((line=reader.readLine())!=null){
                String s[]=line.split("!!!");
                mapDistinct_authID_authName.put(new Integer(s[1]) , s[0]);
                mapDistinct_authID_sourceChannel.put(new Integer(s[1]) , s[2]);
            }
            
            reader=new BufferedReader(new FileReader(new File(inFile_1_output_file_plsi) ));
            TreeMap<Integer, TreeMap<Integer, Double>> map_each_docID_topic_id_prob_score=
            				new TreeMap<Integer, TreeMap<Integer, Double>>();
             int lineNumber=0;
             int last_doc_id=-1;
             boolean error_exist=false; int no_not_found_doc=0;
             TreeMap<Integer,Double> currDocIDs_topic_id_prob_score=new TreeMap<Integer,Double>();
            // each Line <d z p_z_d>   => <0 0 1.903989735839671E-5>
            while((line=reader.readLine())!=null){
            	lineNumber++;
                String s[]=line.split(" ");
                if(lineNumber==1) continue; //header
//                System.out.println("line:"+line);
                int curr_doc_id=Integer.valueOf(s[0]); // doc_id starts with zero , so add 1 (as our doc_id starts with 1 in other programs)
                Integer topic_id= Integer.valueOf(s[1]); //
                Double prob=Double.valueOf(s[2]);
                //String [] arr_topic_and_prob_string=s[2].split(" ");
                
//                System.out.println("topic_id:"+topic_id+" prob:"+prob+" curr_doc_id:"+curr_doc_id);
                
                //if(curr_doc_id>1) break; //fast debug
                
                // topic_id , probability
                
                if(!mapDistinct_topicID_as_KEY.containsKey(topic_id))
                	mapDistinct_topicID_as_KEY.put(topic_id, -1); 
                //
                if(last_doc_id>0 && last_doc_id!=curr_doc_id){
                	map_each_docID_topic_id_prob_score.put(last_doc_id, currDocIDs_topic_id_prob_score);
                	 
                	writerDebug.append("\ndocID="+last_doc_id+" topicID:prob="+currDocIDs_topic_id_prob_score);
                	
                	writerDebug.flush();
                	currDocIDs_topic_id_prob_score=new TreeMap<Integer,Double>();
                }
                //
                currDocIDs_topic_id_prob_score.put(topic_id, prob);
                last_doc_id=curr_doc_id;
                
            }// end while
             
            map_each_docID_topic_id_prob_score.put(last_doc_id, currDocIDs_topic_id_prob_score);
            
            writerDebug.append("\n map_each_docID_topic_id_prob_score.size:"+map_each_docID_topic_id_prob_score.size());
            writerDebug.flush();
            
            reader=new BufferedReader(new FileReader(new File(inFile_2_authorID_docIDCSV) ));
            TreeMap<Integer, TreeMap<Integer, Double>> map_each_authID_topic_id_CUMM_prob_score
            					=new TreeMap<Integer, TreeMap<Integer, Double>>();
            int lineNo=0;
            try{
            // read "!!!authid!!!doc_IDCSV" <- first blank
            // each Line
            while((line=reader.readLine())!=null){
            	lineNo++;
                TreeMap<Integer, Double> currAuth_topicID_CUMM_score=new TreeMap<Integer, Double>();
                String s[]=line.split("!!!");
                int curr_auth_id=Integer.valueOf(s[1]);
                String curr_docID_CSV=s[2];
                String [] arr_curr_docID=curr_docID_CSV.split(",");
                int curr_auth_total_docs_count=arr_curr_docID.length;
                int k=0;
                TreeMap<Integer, Double> curr_authors_curr_doc_topic_id_prob_score=new TreeMap<Integer, Double>();
                
                mapDistinct_authID_as_KEY.put(curr_auth_id, -1);
                
                // curr author , process each doc
                while(k < curr_auth_total_docs_count){
                	
                    // NOT found doc ID
                	if(!map_each_docID_topic_id_prob_score.containsKey(Integer.valueOf(arr_curr_docID[k]))){
	                    writerDebug.append("\n doc ID not found:"+arr_curr_docID[k]);
	                    writerDebug.flush();
	                    error_exist=true;
	                    no_not_found_doc++;
	                    k++;
	                    continue;
                	}
                	else{
                		System.out.println("\n doc ID found:"+arr_curr_docID[k]);
                		writerDebug.append("\n doc ID   found:"+arr_curr_docID[k]);
                	}
                	
                    curr_authors_curr_doc_topic_id_prob_score=
                    		map_each_docID_topic_id_prob_score.get(Integer.valueOf(arr_curr_docID[k]));
                    
                    System.out.println("curr_authors_curr_doc_topic_id_prob_score:"+curr_authors_curr_doc_topic_id_prob_score
                    									+" for doc id:"+arr_curr_docID[k]);
                    
                    //iterate over each possible topic ID and its probability
                    for(int topic_id:mapDistinct_topicID_as_KEY.keySet()){
                    	
                    	
                    	try{
	                        //
	                        if(!currAuth_topicID_CUMM_score.containsKey(topic_id)){
	                            currAuth_topicID_CUMM_score.put(topic_id,
	                                                            curr_authors_curr_doc_topic_id_prob_score.get(topic_id)
	                                                            ); //CUMM_score
	                            System.out.println("cumm topic_id score.1:"+topic_id+"  lineNo="+ lineNo);
	                           
	                        }
	                        else{
	                        
	                            Double new_score=currAuth_topicID_CUMM_score.get(topic_id)+
	                                             curr_authors_curr_doc_topic_id_prob_score.get(topic_id);
	                            currAuth_topicID_CUMM_score.put(topic_id, new_score); //CUMM_score
	                            System.out.println("CUMM cumm topic_id score.2:"+topic_id+"  lineNo="+ lineNo);
	                        }
                        
                    	}
                    	catch(Exception e){
                    		System.out.println("ERROR:getting curr author, curr doc, curr topic id, score? "+topic_id);
                    	}
                    }
                    k++;
                } // end while
                // writing curr_auth CUMM SCORE for each topic_id
               
                map_each_authID_topic_id_CUMM_prob_score.put(curr_auth_id, currAuth_topicID_CUMM_score);
                System.out.println(  "curr_auth_id:"+curr_auth_id
                					+" currAuth_topicID_CUMM_score:"+currAuth_topicID_CUMM_score);
               
//                writer.append(curr_auth_id+"!!!"+currAuth_topicID_CUMM_score+"\n");
//                writer.flush();
                

            }
            
            writerDebug.append("..processed lines : lineNo:"+lineNo+" from file:"+inFile_2_authorID_docIDCSV+"\n");
            writerDebug.flush();
            
            }
            catch(Exception e){
            	System.out.println("\nerror: "+e.getMessage());
            	e.printStackTrace();
            	writerDebug.append("\nerror: "+e.getMessage());
            	writerDebug.flush();
            }
            System.out.println("END: map_each_authID_topic_id_CUMM_prob_score:"+map_each_authID_topic_id_CUMM_prob_score);
            System.out.println("map_each_docID_topic_id_prob_score(size):"+map_each_docID_topic_id_prob_score.size());
            //writerDebug.append("\nmap_each_docID_topic_id_prob_score:"+map_each_docID_topic_id_prob_score);
            writerDebug.flush();
            writer.append("auth_id!!!Prob!!!topic_id!!!authName!!!z_q_topic_manual!!!sourceChannel\n");
            writer.flush();
            writer_TRAD.append("auth_id!!!Prob!!!topic_id!!!authName!!!z_q_topic_manual!!!sourceChannel\n");
            writer_TRAD.flush();
            writer_POLI.append("auth_id!!!Prob!!!topic_id!!!authName!!!z_q_topic_manual!!!sourceChannel\n");
            writer_POLI.flush();
            
            // get Top N authors for each Q topic ( = latent topic in LDA )
            // For each topic , get <authID, CUMM_topic_score>
            for(int curr_topic_id:mapDistinct_topicID_as_KEY.keySet()){
            	TreeMap<Integer, Double> map_AuthID_topicProb=new TreeMap<Integer, Double>();
            	TreeMap<Integer,Double> map_authID_topicProb_sorted= new TreeMap<Integer, Double>();
            	// DEBUG
            	writerDebug.append("curr_topic_id:"+curr_topic_id+" mapDistinct_authID_as_KEY.size:"+mapDistinct_authID_as_KEY.size()+"\n");
            	writerDebug.flush();
            	
            	//
            	for(int curr_authID:mapDistinct_authID_as_KEY.keySet()){
            		
            		
            		if(map_each_authID_topic_id_CUMM_prob_score.containsKey(curr_authID)){
	            		if(map_each_authID_topic_id_CUMM_prob_score.get(curr_authID).containsKey(curr_topic_id)  ){
	            			
	            			if(map_each_authID_topic_id_CUMM_prob_score.get(curr_authID).get(curr_topic_id)!=null)
		            		map_AuthID_topicProb.put(curr_authID,
		            					map_each_authID_topic_id_CUMM_prob_score.get(curr_authID).get(curr_topic_id) );
		            		
//		            	//            		map_AuthID_topicProb.put(curr_authID,
//							map_each_authID_topic_id_CUMM_prob_score.get(curr_authID).get(curr_topic_id));	
//		               		System.out.println("map_each_authID_topic_id_CUMM_prob_score.size:"
//        							+map_each_authID_topic_id_CUMM_prob_score.size());
//		        		System.out.println("curr_authID:"+curr_authID+" map_each_authID_topic_id_CUMM_prob_score.get(authID):"+
//											map_each_authID_topic_id_CUMM_prob_score.get(curr_authID));
//		        		System.out.println("curr_topic_id:"+curr_topic_id
//		        						    +" map_each_authID_topic_id_CUMM_prob_score.get(curr_authID).get(curr_topic_id):"
//											+map_each_authID_topic_id_CUMM_prob_score.get(curr_authID).get(curr_topic_id));
		            		
	            		}
            		}
            		
     

            	}
            	//
            	writerDebug.append("\n mapDistinct_topicID_as_KEY.size:"+mapDistinct_topicID_as_KEY.size()+"<-->"+mapDistinct_topicID_as_KEY);
            	writerDebug.flush();
            	
            	System.out.println("map_AuthID_topicProb:"+map_AuthID_topicProb);
            	
            	if(map_AuthID_topicProb==null ||  map_AuthID_topicProb.size()==0){
            		writerDebug.append("\n map_AuthID_topicProb: NULL : curr_topic_id "+curr_topic_id+"\n");
            		writerDebug.flush();
            	}
            	
            	
            	map_authID_topicProb_sorted=(TreeMap<Integer, Double>) 
            								sort_given_treemap.sortByValue(map_AuthID_topicProb);
            	int c=0; int c_P=0; int c_T=0;
            	for(int auth_id:map_authID_topicProb_sorted.keySet() ){
					System.out.println(auth_id+"!!!"+map_authID_topicProb_sorted.get(auth_id));
					
					String newLine=auth_id
							  +"!!!"+map_authID_topicProb_sorted.get(auth_id)
							  +"!!!"+curr_topic_id
							  +"!!!"+mapDistinct_authID_authName.get(auth_id)
							  +"!!!"+z_q_topic_manually_found[curr_topic_id]+"("+c+")"
							  +"!!!"+mapDistinct_authID_sourceChannel.get(auth_id)
							  +"\n";
					
					writer.append(newLine);
					writer.flush();
					
					
					if(newLine.indexOf(pattern_for_sourceChannel_TRAD)>=0 ){
						c_T++;
						
						if(!map_authID_minAuthId_inFile10.containsKey(auth_id)){
							newLine=		 auth_id
									  +"!!!"+map_authID_topicProb_sorted.get(auth_id)
									  +"!!!"+curr_topic_id
									  +"!!!"+mapDistinct_authID_authName.get(auth_id)
									  +"!!!"+z_q_topic_manually_found[curr_topic_id]+"("+c_T+")"
									  +"!!!"+mapDistinct_authID_sourceChannel.get(auth_id)
									  +"\n";
						}
						else{ //APPLYING MIN AUTHID
							newLine=        map_authID_minAuthId_inFile10.get(auth_id)
									  +"!!!"+map_authID_topicProb_sorted.get(auth_id)
									  +"!!!"+curr_topic_id
									  +"!!!"+mapDistinct_authID_authName.get(auth_id)
									  +"!!!"+z_q_topic_manually_found[curr_topic_id]+"("+c_T+")"
									  +"!!!"+mapDistinct_authID_sourceChannel.get(auth_id)
									  +"\n";
							
						}
						
						
						writer_TRAD.append(newLine);
						writer_TRAD.flush();
					}
					if(newLine.indexOf(pattern_for_sourceChannel_POLI)>=0 ){
						c_P++;
						// 
						if(!map_authID_minAuthId_inFile10.containsKey(auth_id)){
								newLine=		auth_id
										  +"!!!"+map_authID_topicProb_sorted.get(auth_id)
										  +"!!!"+curr_topic_id
										  +"!!!"+mapDistinct_authID_authName.get(auth_id)
										  +"!!!"+z_q_topic_manually_found[curr_topic_id]+"("+c_P+")"
										  +"!!!"+mapDistinct_authID_sourceChannel.get(auth_id)
										  +"\n";
						}
						else{ //APPLYING MIN AUTHID
							newLine=		map_authID_minAuthId_inFile10.get(auth_id)
									  +"!!!"+map_authID_topicProb_sorted.get(auth_id)
									  +"!!!"+curr_topic_id
									  +"!!!"+mapDistinct_authID_authName.get(auth_id)
									  +"!!!"+z_q_topic_manually_found[curr_topic_id]+"("+c_P+")"
									  +"!!!"+mapDistinct_authID_sourceChannel.get(auth_id)
									  +"\n";
							
						}
						
						writer_POLI.append(newLine);
						writer_POLI.flush();
					}
					
					c++;
				}
            	
            	System.out.println("(sort)curr_topic_id:"+curr_topic_id
            						+" map_AuthID_topicProb.size:"+map_AuthID_topicProb.size());
                
            }
            
            writerDebug.flush();
             
            
            System.out.println("mapDistinct_topicID_as_KEY.SIZE:"+mapDistinct_topicID_as_KEY.size());
            
            if(error_exist==true)
            	System.out.println("*** some error: missing doc id");
            else {
            	System.out.println("NO error: NO missing doc id");
            }
            System.out.println("Num of not found document:"+no_not_found_doc);
            //System.out.println("map_each_docID_topic_id_prob_score:"+map_each_docID_topic_id_prob_score);
            
            //writerDebug.append("\nmap_each_docID_topic_id_prob_score:"+map_each_docID_topic_id_prob_score);
            writerDebug.flush();
        }
        catch(Exception e){
            e.printStackTrace();
        }
        return mapAuthId_unique_old_new;
    }
    // main
    public static void main(String[] args) {
    	// (1) RUN "RunPlsi.java" and get output "PLSI-15-T8-.d_by_z.txt"
    	// (2) Update variable "latent_topic_manuallyFound_CSV" in this script.
    	// (3) Approach 1: (a) Run "p6_ground_truth_plsi.java"... (b) run ground_truth.java with Flag==34 after updating "latent_topic_manuallyFound_CSV"
    	// (4) Approach 2: either rename "PLSI-15-T8-.d_by_z.txt" into "LDA_d_by_z.txt." Copy this "LDA_d_by_z.txt" into <ds10> folder 
    	//	   and Run ground_truth.java with Flag==34
    	
        String baseFolder="/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/";

        String folder_prefix_LDA_or_PLSI="LDA";
        String baseFolder_PLSI_or_LDA=baseFolder+ folder_prefix_LDA_or_PLSI+ "/"+folder_prefix_LDA_or_PLSI+"_run1/";
        // INPUT
        String inFile_1_output_file_plsi=baseFolder_PLSI_or_LDA+ "PLSI-15-T8-.d_by_z.txt"; // "LDA_d_by_z_prob.txt"; //INPUT
        	   inFile_1_output_file_plsi=baseFolder_PLSI_or_LDA+"LDA_d_by_z_prob.txt";
        String inFile_2_authorID_docIDCSV=baseFolder+"auth_id_doc_id.txt_NO_dirtyAuthNames.txt"; //INPUT
        String inFile_3_authorName_AuthID=baseFolder+"authName_AND_auth_id.txt_NO_dirtyAuthNames.txt"; //INPUT
        
        //OUTPUT file
        String outFile_each_auth_CUMM_topic_score=baseFolder_PLSI_or_LDA+ folder_prefix_LDA_or_PLSI+ "_outFile_each_auth_CUMM_topic_score.txt";
 
        //from z=0 to 9 (manually marked for each clusters from z=0 to 9)
        String latent_topic_manuallyFound_CSV="syria,crime,india,climate change,election,boko haram";
        	   latent_topic_manuallyFound_CSV="crime,climate change,india,election,syria,boko haram";  //attempt 2
               latent_topic_manuallyFound_CSV="climate change,boko haram,syria,india,crime,election";  //attempt 1
               latent_topic_manuallyFound_CSV="syria,election,climate change,crime,boko haram,india";  //attempt 3
               latent_topic_manuallyFound_CSV="syria,election,india,crime,boko haram,climate change";//attempt 4
               latent_topic_manuallyFound_CSV="boko haram,election,climate change,crime,india,syria"; // PAM output
        //cancer, traffick
        String outFile_each_auth_CUMM_topic_score_POLI=baseFolder_PLSI_or_LDA+ folder_prefix_LDA_or_PLSI+ "_outFile_each_auth_CUMM_topic_score_POLI.txt";
        String outFile_each_auth_CUMM_topic_score_TRAD=baseFolder_PLSI_or_LDA+ folder_prefix_LDA_or_PLSI+ "_outFile_each_auth_CUMM_topic_score_TRAD.txt";
        //"/Users/lenin/Downloads/p6/merged/dummy.mergeall/authornameextract2/PLSI/plsi-15-T2-.d_by_z.out";
        
        String inFile_10_authName_AND_auth_id__MAPPEDmultiAuthID_SORTED_add_minAuthID=baseFolder+"authName_AND_auth_id.txt__MAPPEDmultiAuthID_SORTED_add_minAuthID.txt";
        
        String 	pattern_for_sourceChannel_TRAD="trad_";
	    String 	pattern_for_sourceChannel_POLI="poli_";
        
        //NOTE: copy d_by_z_prob file from base folder to LDA folder
        
        // input_authID_pair_out_unique_authID
        read_plsi_output_file_d_z_probability(
			                                        baseFolder,
			                                        inFile_1_output_file_plsi,
			                                        inFile_2_authorID_docIDCSV,
			                                        inFile_3_authorName_AuthID,
			                                        inFile_10_authName_AND_auth_id__MAPPEDmultiAuthID_SORTED_add_minAuthID,
			                                        outFile_each_auth_CUMM_topic_score,
			                                        outFile_each_auth_CUMM_topic_score_TRAD,
			                                        outFile_each_auth_CUMM_topic_score_POLI,
			                                        latent_topic_manuallyFound_CSV,
			                                        pattern_for_sourceChannel_TRAD,
			                                        pattern_for_sourceChannel_POLI
			                                        );
    }

}	
	
	

