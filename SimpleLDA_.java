/* Copyright (C) 2005 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.	For further
   information, see the file `LICENSE' included with this distribution. 
// lenin: refer http://www.jmlr.org/proceedings/papers/v13/xiao10a/xiao10a.pdf
// http://www.ics.uci.edu/~asuncion/pubs/UAI_09.pdf
   */

package cc.mallet.topics;

import cc.mallet.topics.ParallelTopicModel;
import java.util.*;
import java.util.Formatter;
import java.util.logging.*;
import java.util.regex.Pattern;
import java.util.zip.*;

import static java.util.concurrent.TimeUnit.NANOSECONDS;

import java.io.*;
import java.text.NumberFormat;

import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.CharSequenceLowercase;
import cc.mallet.pipe.Input2CharSequence;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SaveDataInSource;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.TokenSequenceNGrams;
import cc.mallet.pipe.TokenSequenceRemoveStopwords;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.pipe.iterator.StringArrayIterator;
import cc.mallet.topics.*;
import cc.mallet.types.*;
import cc.mallet.util.*;
import crawler.print_all_methods_of_a_given_class;
import crawler.readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile;
import crawler.readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.*;

/**
 * A simple implementation of Latent Dirichlet Allocation using Gibbs sampling.
 * This code is slower than the regular Mallet LDA implementation, but provides a 
 *  better starting place for understanding how sampling works and for 
 *  building new topic models.
 * 
 * @author David Mimno, Andrew McCallum
 */

public class SimpleLDA_ implements Serializable {

	private static Logger logger = MalletLogger.getLogger(SimpleLDA_.class.getName());
	
	// the training instances and their topic assignments
	protected ArrayList<TopicAssignment> data;

	// the alphabet for the input data
	protected Alphabet alphabet; 

	// the alphabet for the topics
	protected LabelAlphabet topicAlphabet; 
	
	// The number of topics requested
	protected int numTopics;
	int	num_of_context=6;

	// The size of the vocabulary
	protected int numTypes;

	// Prior parameters
	protected double alpha;	 // Dirichlet(alpha,alpha,...) is the distribution over topics
	protected double alphaSum;
	protected double beta;   // Prior on per-topic multinomial distribution over words
	protected double betaSum;
	public static final double DEFAULT_BETA = 0.01;
	public double gamma;	 // Dirichlet(alpha,alpha,...) is the distribution over topics
	protected double gammaSum;
	
	// An array to put the topic counts for the current document. 
	// Initialized locally below.  Defined here to avoid
	// garbage collection overhead.
	protected int[] oneDocTopicCounts; // indexed by <document index, topic index>

	// Statistics needed for sampling.
	protected int[][] typeTopicCounts; // indexed by <feature index, topic index>
	protected int[] tokensPerTopic; // indexed by <topic index>
	
	public int showTopicsInterval = 50;
	public int wordsPerTopic = 20;
	
	protected static Randoms random;
	protected NumberFormat formatter;
	protected boolean printLogLikelihood = false;
	//CONTEXT
	int[][] typeTopicCounts_curr_CONTEXT=new int[num_of_context][numTopics];
	int[] typeTopicCounts_curr_CONTEXT_onlyTopic;
	int[] tokensPerTopic_for_CONTEXT=new int[numTopics]; // indexed by <topic index>
	// KEYWORDS N MISSINGkeywords
	int[][] typeTopicCounts_KEYWORDS_N_MISSINGkeywords;
	int[][] typeCounts_KEYWORDS_N_MISSINGkeywords;
	
	/// TITLE
	// the alphabet for the input data
	protected Alphabet alphabet_TITLE; 
	// the alphabet for the topics
	protected LabelAlphabet topicAlphabet_TITLE;
	// The number of topics requested
	protected int numTopics_TITLE;
	// The size of the vocabulary
	protected int numTypes_TITLE;
	protected int[] oneDocTopicCounts_TITLE; // indexed by <document index, topic index>
	int[][] typeTopicCounts_TITLE; // indexed by <feature index, topic index>
	int[] tokensPerTopic_TITLE=new int[numTopics]; // indexed by <topic index>
	
	int[][] typeCounts_TITLE;
	
	// PERSON_LOCATION
	protected Alphabet alphabet_PERSON_LOCATION; 
	// the alphabet for the topics
	protected LabelAlphabet topicAlphabet_PERSON_LOCATION;
	// The number of topics requested
	protected int numTopics_PERSON_LOCATION;
	// The size of the vocabulary
	protected int numTypes_PERSON_LOCATION;
	protected int[] oneDocTopicCounts_PERSON_LOCATION; // indexed by <document index, topic index>
	int[][] typeTopicCounts_PERSON_LOCATION; // indexed by <feature index, topic index>
	int[] tokensPerTopic_PERSON_LOCATION=new int[numTopics]; // indexed by <topic index>
	
	int[][] typeCounts_PERSON_LOCATION;
	
	// KEYWORDS
	protected Alphabet alphabet_KEYWORDS; 
	// the alphabet for the topics
	protected LabelAlphabet topicAlphabet_KEYWORDS;
	// The number of topics requested
	protected int numTopics_KEYWORDS;
	// The size of the vocabulary
	protected int numTypes_KEYWORDS;
	protected int[] oneDocTopicCounts_KEYWORDS; // indexed by <document index, topic index>
	int[][] typeTopicCounts_KEYWORDS; // indexed by <feature index, topic index>
	int[] tokensPerTopic_KEYWORDS=new int[numTopics]; // indexed by <topic index>
	int[][] typeCounts_KEYWORDS;
	
	// FIRST_LINE
	protected Alphabet alphabet_FIRST_LINE; 
	// the alphabet for the topics
	protected LabelAlphabet topicAlphabet_FIRST_LINE;
	// The number of topics requested
	protected int numTopics_FIRST_LINE;
	// The size of the vocabulary
	protected int numTypes_FIRST_LINE;
	protected int[] oneDocTopicCounts_FIRST_LINE; // indexed by <document index, topic index>
	int[][] typeTopicCounts_FIRST_LINE; // indexed by <feature index, topic index>
	int[] tokensPerTopic_FIRST_LINE=new int[numTopics]; // indexed by <topic index>
	
	int[][] typeCounts_FIRST_LINE;
	
	// ORGANIZATION
	protected Alphabet alphabet_ORGANIZATION; 
	// the alphabet for the topics
	protected LabelAlphabet topicAlphabet_ORGANIZATION;
	// The number of topics requested
	protected int numTopics_ORGANIZATION;
	// The size of the vocabulary
	protected int numTypes_ORGANIZATION;
	protected int[] oneDocTopicCounts_ORGANIZATION; // indexed by <document index, topic index>
	int[][] typeTopicCounts_ORGANIZATION; // indexed by <feature index, topic index>
	int[] tokensPerTopic_ORGANIZATION=new int[numTopics]; // indexed by <topic index>
	
	int[][] typeCounts_ORGANIZATION;
	
	// NUMBERS
	protected Alphabet alphabet_NUMBERS; 
	// the alphabet for the topics
	protected LabelAlphabet topicAlphabet_NUMBERS;
	// The number of topics requested
	protected int numTopics_NUMBERS;
	// The size of the vocabulary
	protected int numTypes_NUMBERS;
	protected int[] oneDocTopicCounts_NUMBERS; // indexed by <document index, topic index>
	int[][] typeTopicCounts_NUMBERS; // indexed by <feature index, topic index>
	int[] tokensPerTopic_NUMBERS=new int[numTopics]; // indexed by <topic index>
	int[][] typeCounts_NUMBERS;
	
	// MISSING_KEYWORDS
	protected Alphabet alphabet_MISSING_KEYWORDS; 
	// the alphabet for the topics
	protected LabelAlphabet topicAlphabet_MISSING_KEYWORDS;
	// The number of topics requested
	protected int numTopics_MISSING_KEYWORDS;
	// The size of the vocabulary
	protected int numTypes_MISSING_KEYWORDS;
	protected int[] oneDocTopicCounts_MISSING_KEYWORDS; // indexed by <document index, topic index>
	int[][] typeTopicCounts_MISSING_KEYWORDS; // indexed by <feature index, topic index>
	int[] tokensPerTopic_MISSING_KEYWORDS=new int[numTopics]; // indexed by <topic index>
	int[][] typeCounts_MISSING_KEYWORDS;
	
	// bool_TRADorPOLI
	protected Alphabet alphabet_bool_TRADorPOLI; 
	// the alphabet for the topics
	protected LabelAlphabet topicAlphabet_bool_TRADorPOLI;
	// The number of topics requested
	protected int numTopics_bool_TRADorPOLI;
	// The size of the vocabulary
	protected int numTypes_bool_TRADorPOLI;
	protected int[] oneDocTopicCounts_bool_TRADorPOLI; // indexed by <document index, topic index>
	int[][] typeTopicCounts_bool_TRADorPOLI; // indexed by <feature index, topic index>
	int[] tokensPerTopic_bool_TRADorPOLI=new int[numTopics]; // indexed by <topic index>
	
	
	/////////LENGTH OF DICTIONARY
	int dict_len_TITLE=0;
	int dict_len_PERSON_LOCATION=0; // person only
	int dict_len_KEYWORDS_missingKEYWORDS=0;
	int dict_len_FIRST_LINE=0;
	int dict_len_ORGANIZATION=0;
	int dict_len_KEYWORDS=0;
	int dict_len_NUMBERS=0; //location only
	 
	
	public int[][] tokensPerTopic_CONTEXT; //=new int[num_of_context][numTopics]; // [context][topic]
	
	//convert_int_array_to_String
	public static String convert_double_array_to_String_print(double[] alpha2){
		
		String result = "";
		for (double s : alpha2) {
		    result =result+ " "+  String.valueOf(s) ;
		}
		return result;
	}
	// 
	public static String convert_int_array_to_String_print(int[] alpha2){		
		String result = "";
		for (int s : alpha2) {
		    result =result+ " "+  String.valueOf(s) ;
		}
		return result;
	}
	// 
	public SimpleLDA_ (int numberOfTopics, int gamma) {
		this (numberOfTopics, numberOfTopics, gamma, DEFAULT_BETA);
	}
	
	public SimpleLDA_(int numberOfTopics, double alphaSum, double beta, double gamma) {
		this (numberOfTopics, alphaSum, beta, gamma,new Randoms());
	}
	
	private static LabelAlphabet newLabelAlphabet (int numTopics) {
		LabelAlphabet ret = new LabelAlphabet();
		for (int i = 0; i < numTopics; i++)
			ret.lookupIndex("topic"+i);
		return ret;
	}
	
	public SimpleLDA_ (int numberOfTopics, double alphaSum, double beta,  double gamma, Randoms random) {
		this (newLabelAlphabet (numberOfTopics), alphaSum, beta, gamma, random);
	}
	
	public SimpleLDA_(LabelAlphabet topicAlphabet, double alphaSum, double beta, double gamma, Randoms random ){
		System.out.println( "****Inside SimpleLDA, setting data, topicAlphabet, numTopics, oneDocTopicCounts, tokensPerTopic.." );
		this.data = new ArrayList<TopicAssignment>();
		this.topicAlphabet = topicAlphabet;
		this.numTopics = topicAlphabet.size();

		this.alphaSum = alphaSum;
		this.alpha = alphaSum / numTopics;
		this.beta = beta;
		this.random = random;
		this.gamma = gamma;
		// BODYTEXT
		oneDocTopicCounts = new int[numTopics];
		tokensPerTopic = new int[numTopics];
		// TITLE
		oneDocTopicCounts_TITLE = new int[numTopics];
		tokensPerTopic_TITLE = new int[numTopics];
		// PERSON
		oneDocTopicCounts_PERSON_LOCATION = new int[numTopics];
		tokensPerTopic_PERSON_LOCATION = new int[numTopics];
		// ORGANIZATION
		oneDocTopicCounts_ORGANIZATION = new int[numTopics];
		tokensPerTopic_ORGANIZATION = new int[numTopics];
		//LOCATION
		oneDocTopicCounts_NUMBERS = new int[numTopics];
		tokensPerTopic_NUMBERS = new int[numTopics];
		// KEYWORDS
		oneDocTopicCounts_MISSING_KEYWORDS = new int[numTopics];
		tokensPerTopic_MISSING_KEYWORDS = new int[numTopics];
		// FIRST LINE
		oneDocTopicCounts_FIRST_LINE = new int[numTopics];
		tokensPerTopic_FIRST_LINE = new int[numTopics];
		// BOOL TRADorPOLI
		oneDocTopicCounts_bool_TRADorPOLI = new int[numTopics];
		tokensPerTopic_bool_TRADorPOLI = new int[numTopics];

		formatter = NumberFormat.getInstance();
		formatter.setMaximumFractionDigits(5);

		
		logger.info("Simple LDA: " + numTopics + " topics");
	}
	
	public Alphabet getAlphabet() { return alphabet; }
	public LabelAlphabet getTopicAlphabet() { return topicAlphabet; }
	public int getNumTopics() { return numTopics; }
	public ArrayList<TopicAssignment> getData() { return data; }
	
	public void setTopicDisplay(int interval, int n) {
		this.showTopicsInterval = interval;
		this.wordsPerTopic = n;
	}

	public void setRandomSeed(int seed) {
		random = new Randoms(seed);
	}
	
	public int[][] getTypeTopicCounts() { return typeTopicCounts; }
	public int[] getTopicTotals() { return tokensPerTopic; }

	public double[] getTopicProbabilities(int instanceID) {
//		System.out.println("inside method getTopicProbabilities");
		LabelSequence topics = data.get(instanceID).topicSequence; //inside getTopicProbabilities()
		return getTopicProbabilities(topics);
	}
	public double[] getTopicProbabilities_TITLE(int instanceID) {
//		System.out.println("inside method getTopicProbabilities");
		LabelSequence topics = data.get(instanceID).topicSequence_title; //inside getTopicProbabilities()
		return getTopicProbabilities_TITLE(topics);
	}
	/** Get the smoothed distribution over topics for a topic sequence, 
	 * which may be from the training set or from a new instance with topics
	 * assigned by an inferencer.
	 */
	public double[] getTopicProbabilities(LabelSequence topics) { //BODYTEXT
// 		System.out.println("------------ Inside method getTopicProbabilities()....");
		double[] topicDistribution = new double[numTopics];
		// Loop over the tokens in the document, counting the current topic
		//  assignments.
		for (int position = 0; position < topics.getLength(); position++) {
			topicDistribution[ topics.getIndexAtPosition(position) ]++;
		}

		// Add the smoothing parameters and normalize
		double sum = 0.0;
		for (int topic = 0; topic < numTopics; topic++) {
//			topicDistribution[topic] += alpha[topic];
			sum += topicDistribution[topic];
		}

		// And normalize
		for (int topic = 0; topic < numTopics; topic++) {
			topicDistribution[topic] /= sum;
		}

		return topicDistribution;
	}
	//TITLE
	public double[] getTopicProbabilities_TITLE(LabelSequence topics) { //TITLE
// 		System.out.println("------------ Inside method getTopicProbabilities()....");
		double[] topicDistribution = new double[numTopics];
		// Loop over the tokens in the document, counting the current topic
		//  assignments.
		for (int position = 0; position < topics.getLength(); position++) {
			topicDistribution[ topics.getIndexAtPosition(position) ]++;
		}
		// Add the smoothing parameters and normalize
		double sum = 0.0;
		for (int topic = 0; topic < numTopics; topic++) {
//			topicDistribution[topic] += alpha[topic];
			sum += topicDistribution[topic];
		}

		// And normalize
		for (int topic = 0; topic < numTopics; topic++) {
			topicDistribution[topic] /= sum;
		}

		return topicDistribution;
	}
	
	public void addInstances (
							 InstanceList training, // bodyText
							 InstanceList instances_numbers,
							 InstanceList instances_person_location,
							 InstanceList instances_organization,
							 InstanceList instances_fline,
							 InstanceList instances_title,
							 InstanceList instances_keywords,
							 InstanceList instances_MISSINGkeywords,
							 InstanceList instances_bool_TRADorPOLI,
							 InstanceList instances_all_features,
							 InstanceList instances_keywords_N__MISSINGkeywords, 
							 FileWriter   writerDebug
//							 InstanceList training2
							) {
		
		try{
		System.out.println("****Inside addInstances (InstanceList training) setting alphabet, numTypes, typeTopicCounts, FeatureSequence, LabelSequence, data.add()");
		
		
		
		//bodyTEXT
		alphabet = training.getDataAlphabet();
		numTypes = alphabet.size();
		betaSum = beta * numTypes;
		gammaSum = gamma* numTypes; //
		typeTopicCounts = new int[numTypes][numTopics];
		//TITLE
		alphabet_TITLE = instances_title.getDataAlphabet();
		numTypes_TITLE = alphabet_TITLE.size();
		typeTopicCounts_TITLE = new int[numTypes_TITLE][numTopics];
		
		tokensPerTopic_KEYWORDS=new int[numTopics];
		tokensPerTopic_NUMBERS=new int[numTopics];
		tokensPerTopic_ORGANIZATION=new int[numTopics];
		tokensPerTopic_PERSON_LOCATION=new int[numTopics];
		tokensPerTopic_FIRST_LINE=new int[numTopics];
		tokensPerTopic_bool_TRADorPOLI=new int[numTopics];
		
		typeTopicCounts_PERSON_LOCATION=new int[200000][numTopics];
		typeTopicCounts_ORGANIZATION=new int[200000][numTopics];
		typeTopicCounts_NUMBERS=new int[200000][numTopics];
		typeTopicCounts_KEYWORDS=new int[200000][numTopics];
		typeTopicCounts_FIRST_LINE=new int[200000][numTopics];
		typeTopicCounts_TITLE=new int[200000][numTopics];
		typeTopicCounts_bool_TRADorPOLI=new int[200000][numTopics];
		typeTopicCounts_curr_CONTEXT=new int[200000][numTopics];
		typeTopicCounts_curr_CONTEXT_onlyTopic=new int[numTopics];
		typeTopicCounts_KEYWORDS_N_MISSINGkeywords=new int[200000][numTopics];
		
		//typeCount
		typeCounts_PERSON_LOCATION=new int[200000][dict_len_PERSON_LOCATION];
		typeCounts_ORGANIZATION=new int[200000][dict_len_ORGANIZATION];
		typeCounts_NUMBERS=new int[200000][dict_len_NUMBERS];
		typeCounts_KEYWORDS=new int[200000][dict_len_KEYWORDS];
		typeCounts_FIRST_LINE=new int[200000][dict_len_FIRST_LINE];
		typeCounts_TITLE=new int[200000][dict_len_TITLE];
		typeCounts_KEYWORDS_N_MISSINGkeywords=new int[200000][dict_len_KEYWORDS_missingKEYWORDS];
		// initialize
		tokensPerTopic_CONTEXT=new int[num_of_context][numTopics];
		
		int doc = 0; int count_instances=0;

		for (Instance instance : training) {
			
			// BEGIN lenin add
			String instance_name=instance.getName().toString();
			count_instances++;
			
			boolean isSOPprint=instance.get_isSOPpring() ;
			
			if(count_instances<=10) //
				System.out.println("------------------------------------------------------------------"+instance_name
									+" count_instances:"+count_instances);
			
			if(isSOPprint){
			System.out.println("-------start search_by_instanceName--> title,pers+locat,keywords,fline,organization,numbers-----");
			System.out.println("----------- title-------");}
			Instance instance_title=instances_title.search_by_instanceName(instance_name, instances_title);
			if(isSOPprint)
				System.out.println("----------- person-------");
			Instance instance_person_location=instances_person_location.search_by_instanceName(instance_name, instances_person_location);
			if(isSOPprint)
				System.out.println("----------- keywords-------");
			Instance instance_keywords=instances_keywords.search_by_instanceName(instance_name, instances_keywords);
			if(isSOPprint)
				System.out.println("----------- fline-------");
			Instance instance_fline=instances_fline.search_by_instanceName(instance_name, instances_fline);
			if(isSOPprint)
				System.out.println("----------- organization-------");
			Instance instance_organization=instances_organization.search_by_instanceName(instance_name, instances_organization);
			if(isSOPprint)
				System.out.println("----------- location-------");
			Instance instance_numbers=instances_numbers.search_by_instanceName(instance_name, instances_numbers);
			if(isSOPprint)
				System.out.println("----------- missingKEYWORDS-------");
			Instance instance_MISSINGkeywords=instances_MISSINGkeywords.search_by_instanceName(instance_name, instances_MISSINGkeywords);
			if(isSOPprint)
				System.out.println("----------- bool_TRADorPOLI-------");
			Instance instance_bool_TRADorPOLI=instances_bool_TRADorPOLI.search_by_instanceName(instance_name, instances_bool_TRADorPOLI);
			if(isSOPprint)
				System.out.println("----------- all_features-------");
			Instance instance_all_features= instances_all_features.search_by_instanceName(instance_name, instances_all_features);
			if(isSOPprint)
				System.out.println("----------- keywords N missingKEYWORDS-------");
			Instance instance_keywords_N_missingKEYWORDS=instances_keywords_N__MISSINGkeywords.search_by_instanceName(instance_name, instances_keywords_N__MISSINGkeywords);
			 
			if(isSOPprint)
				System.out.println("-------end search_by_instanceName--> ");
			
			// END lenin add
			
			if(doc<=10 || doc>9000){ //debug
				System.out.println("doc:"+doc+" length of dictionary(contexts):"+instances_title.getDataAlphabet().size()
											+" "+instance_person_location.getDataAlphabet().size()
											+" "+instance_organization.getDataAlphabet().size()
											  +" "+instance_MISSINGkeywords.getDataAlphabet().size()
											  +" "+instances_fline.getDataAlphabet().size()); //this gives size of
				dict_len_FIRST_LINE=instances_fline.getDataAlphabet().size();
				dict_len_KEYWORDS_missingKEYWORDS=instance_MISSINGkeywords.getDataAlphabet().size();
				dict_len_ORGANIZATION=instance_organization.getDataAlphabet().size();
				dict_len_TITLE=instances_title.getDataAlphabet().size();
				dict_len_PERSON_LOCATION=instance_person_location.getDataAlphabet().size();// person only
				dict_len_KEYWORDS=instance_keywords.getDataAlphabet().size();
				dict_len_NUMBERS=instance_numbers.getDataAlphabet().size();
			}
			
			doc++;
			//bodyTEXT
			FeatureSequence tokens = (FeatureSequence) instance.getData();
			LabelSequence topicSequence = new LabelSequence(topicAlphabet, new int[ tokens.size() ]);
			
			//TITLE
			FeatureSequence tokens_title = (FeatureSequence) instance_title.getData(); //note this get data - inside addInstances() //lenin
			LabelSequence topicSequence_title =new LabelSequence(topicAlphabet, new int[ tokens_title.size() ]);// inside addInstances()
			// PERSON 
			FeatureSequence tokenSequence_person_location =
						(FeatureSequence) instance_person_location.getData(); // inside run() -- WorkerRunnable_p6.javaLabelSequence topicSequence =
			LabelSequence topicSequence_person_location =
						new LabelSequence(topicAlphabet, new int[ tokenSequence_person_location.size() ]);// inside run()  -- WorkerRunnable_p6.java
			// KEYWORDS 
			FeatureSequence tokenSequence_keywords =
					(FeatureSequence) instance_keywords.getData(); // inside run() -- WorkerRunnable_p6.javaLabelSequence topicSequence =
			LabelSequence topicSequence_keywords =
					new LabelSequence(topicAlphabet, new int[ tokenSequence_keywords.size() ]);// inside run()  -- WorkerRunnable_p6.java
			
			// FLINE
			FeatureSequence tokenSequence_fline =
					(FeatureSequence) instance_fline.getData(); // inside run() -- WorkerRunnable_p6.javaLabelSequence topicSequence =
			LabelSequence topicSequence_fline =
					new LabelSequence(topicAlphabet, new int[ tokenSequence_fline.size() ]);// inside run()  -- WorkerRunnable_p6.java
			// ORGANIZATION
			FeatureSequence tokenSequence_organization =
					(FeatureSequence) instance_organization.getData(); // inside run() -- WorkerRunnable_p6.javaLabelSequence topicSequence =
			LabelSequence topicSequence_organization =
					new LabelSequence(topicAlphabet, new int[ tokenSequence_organization.size() ]);// inside run()  -- WorkerRunnable_p6.java					
			// LOCATION
			FeatureSequence tokenSequence_numbers =
					(FeatureSequence) instance_numbers.getData(); // inside run() -- WorkerRunnable_p6.javaLabelSequence topicSequence =
			LabelSequence topicSequence_numbers =
					new LabelSequence(topicAlphabet, new int[ tokenSequence_numbers.size() ]);// inside run()  -- WorkerRunnable_p6.java
			
			//MISSINGkeywords
			FeatureSequence tokenSequence_MISSINGkeywords =
					(FeatureSequence) instance_MISSINGkeywords.getData(); // inside run() -- WorkerRunnable_p6.javaLabelSequence topicSequence =
			LabelSequence topicSequence_MISSINGkeywords =
					new LabelSequence(topicAlphabet, new int[ tokenSequence_MISSINGkeywords.size() ]);// inside run()  -- WorkerRunnable_p6.java
			// bool_TRADorPOLI 
			FeatureSequence tokenSequence_bool_TRADorPOLI =
					(FeatureSequence) instance_bool_TRADorPOLI.getData(); // inside run() -- WorkerRunnable_p6.javaLabelSequence topicSequence =
			LabelSequence topicSequence_bool_TRADorPOLI =
					new LabelSequence(topicAlphabet, new int[ tokenSequence_bool_TRADorPOLI.size() ]);// inside run()  -- WorkerRunnable_p6.java
			// ALL FEATURES
			FeatureSequence tokens_all_features = (FeatureSequence) instance_all_features.getData(); //note this get data - inside addInstances() //lenin
			LabelSequence topicSequence_all_features =new LabelSequence(topicAlphabet, new int[ tokens_all_features.size() ]);// inside addInstances()
			
			// ALL FEATURES
			FeatureSequence tokensSequence_keywords_N_missingKEYWORDS = (FeatureSequence) instance_keywords_N_missingKEYWORDS.getData(); //note this get data - inside addInstances() //lenin
			LabelSequence topicSequence_keywords_N_missingKEYWORDS =new LabelSequence(topicAlphabet, new int[ tokensSequence_keywords_N_missingKEYWORDS.size() ]);// inside addInstances()
			
			int[] topics = topicSequence.getFeatures();
			//bodyTEXT
			for (int position = 0; position < tokens.size(); position++) {
				// randomly set the topics ...
				int topic = random.nextInt(numTopics);
				topics[position] = topic;
				tokensPerTopic[topic]++;
				
				int type = tokens.getIndexAtPosition(position);
				typeTopicCounts[type][topic]++;
			} // NOT touched by LM
			
			tokensPerTopic_for_CONTEXT=new int[numTopics]; //LENIN: check on this LATER, why I need to set here.
			
			int itr=100; int cnt20=0;
			FeatureSequence temp_curr_context_tokenSequence=null;
			// intitalize
			for (int context = 0; context < num_of_context; context++) {
				for (int topic100 = 0; topic100 < numTopics; topic100++) {
					if(context==0){
						temp_curr_context_tokenSequence=tokens_title;
						typeTopicCounts_curr_CONTEXT=typeTopicCounts_TITLE;
					}
					if(context==1){
						temp_curr_context_tokenSequence=tokenSequence_person_location;
						typeTopicCounts_curr_CONTEXT=typeTopicCounts_PERSON_LOCATION;//this is person ONLY
					}
					if(context==2){
						temp_curr_context_tokenSequence=tokenSequence_keywords;
						typeTopicCounts_curr_CONTEXT=typeTopicCounts_KEYWORDS_N_MISSINGkeywords; // KEYWORD(applied tags) + missing (related keywords)
//						typeTopicCounts_curr_CONTEXT=typeTopicCounts_KEYWORDS;
					}
					if(context==3){
						temp_curr_context_tokenSequence=tokenSequence_fline;
						typeTopicCounts_curr_CONTEXT=typeTopicCounts_FIRST_LINE;
					}
					if(context==4){ 
						temp_curr_context_tokenSequence=tokenSequence_organization;
						typeTopicCounts_curr_CONTEXT=typeTopicCounts_ORGANIZATION;
					}
					if(context==5){
						temp_curr_context_tokenSequence=tokenSequence_numbers;
						typeTopicCounts_curr_CONTEXT=typeTopicCounts_NUMBERS; //this is location ONLY
					}
 
					// for TEMP CURRENT CONTEXT 
					for (int position22 = 0; position22 < temp_curr_context_tokenSequence.size(); position22++){
						int type = position22 ;
						typeTopicCounts_curr_CONTEXT[type][topic100]=1; //initialize
					}
				}
			}
			
			while(cnt20<itr){
			// CONTEXT
				for (int curr_context = 0; curr_context < num_of_context; curr_context++) {
					// randomly set the topics ...
					int topic = random.nextInt(numTopics);
	//				System.out.println("10.topic:"+topic+" tokensPerTopic_for_CONTEXT.len:"+tokensPerTopic_for_CONTEXT.length);
	//				topics[position] = topic;
//					tokensPerTopic_for_CONTEXT[topic]++;
					
					if(curr_context==0){
						temp_curr_context_tokenSequence=tokens_title;
						typeTopicCounts_curr_CONTEXT=typeTopicCounts_TITLE;
					}
					if(curr_context==1){
						temp_curr_context_tokenSequence=tokenSequence_person_location;
						typeTopicCounts_curr_CONTEXT=typeTopicCounts_PERSON_LOCATION;
					}
					if(curr_context==2){
						temp_curr_context_tokenSequence=tokenSequence_keywords;
						typeTopicCounts_curr_CONTEXT=typeTopicCounts_KEYWORDS_N_MISSINGkeywords; // KEYWORD(applied tags) + missing (related keywords)
//						typeTopicCounts_curr_CONTEXT=typeTopicCounts_KEYWORDS;
					}
					if(curr_context==3){
						temp_curr_context_tokenSequence=tokenSequence_fline;
						typeTopicCounts_curr_CONTEXT=typeTopicCounts_FIRST_LINE;
					}
					if(curr_context==4){ 
						temp_curr_context_tokenSequence=tokenSequence_organization;
						typeTopicCounts_curr_CONTEXT=typeTopicCounts_ORGANIZATION;
					}
					if(curr_context==5){ //LOCATION
						temp_curr_context_tokenSequence=tokenSequence_numbers;
						typeTopicCounts_curr_CONTEXT=typeTopicCounts_NUMBERS; //LOCATION
					}
//					if(curr_context==6){
//						temp_curr_context_tokenSequence=tokenSequence_MISSINGkeywords;
//						typeTopicCounts_curr_CONTEXT=typeTopicCounts_MISSING_KEYWORDS;
//					}
//					if(curr_context==7){
//						temp_curr_context_tokenSequence=tokenSequence_bool_TRADorPOLI;
//						typeTopicCounts_curr_CONTEXT=typeTopicCounts_bool_TRADorPOLI;
//					}
//					System.out.println("cxt:"+curr_context+" "+topic+" "+tokensPerTopic_CONTEXT[0].length+" "+numTopics);
					//number of tokens per topic under each context.. new2
					tokensPerTopic_CONTEXT[curr_context][topic]++;
					
					// for TEMP CURRENT CONTEXT 
					for (int position2 = 0; position2 < temp_curr_context_tokenSequence.size(); position2++){
						int type = position2 ;
						typeTopicCounts_curr_CONTEXT[type][topic]++;
					}
				} //
				cnt20++;
			}
			
			// TITLE // randomly set the topics ...
			int[] topics_TITLE = topicSequence_title.getFeatures();
			
			for (int position = 0; position < tokens_title.size(); position++) {
				// randomly set the topics ...
				int topic = random.nextInt(numTopics);
				topics_TITLE[position] = topic;
//				System.out.println("tokensPerTopic_TITLE:"+tokensPerTopic_TITLE);
				tokensPerTopic_TITLE[topic]++;
				
				int type = tokens_title.getIndexAtPosition(position);
				typeTopicCounts_TITLE[type][topic]++;
			} //
			
			// PERSON // randomly set the topics ...
			int[] topics_PERSON_LOCATION = topicSequence_person_location.getFeatures();
			for (int position = 0; position < tokenSequence_person_location.size(); position++) {
				// randomly set the topics ...
				int topic = random.nextInt(numTopics);
				topics_PERSON_LOCATION[position] = topic;
//				System.out.println("tokensPerTopic_TITLE:"+tokensPerTopic_TITLE);
				tokensPerTopic_PERSON_LOCATION[topic]++;
				
				int type = tokenSequence_person_location.getIndexAtPosition(position);
				typeTopicCounts_PERSON_LOCATION[type][topic]++;
			} // 
			// ORGANIZATION // randomly set the topics ...
			int[] topics_ORGANIZATION = topicSequence_organization.getFeatures();
			for (int position = 0; position < tokenSequence_organization.size(); position++) {
				// randomly set the topics ...
				int topic = random.nextInt(numTopics);
				topics_ORGANIZATION[position] = topic;
//				System.out.println("tokensPerTopic_TITLE:"+tokensPerTopic_TITLE);
				tokensPerTopic_ORGANIZATION[topic]++;
				int type = tokenSequence_organization.getIndexAtPosition(position);
				typeTopicCounts_ORGANIZATION[type][topic]++;
			} // 
			// LOCATION // randomly set the topics ...
			int[] topics_NUMBERS = topicSequence_numbers.getFeatures();
			
			for (int position = 0; position < tokenSequence_numbers.size(); position++) {
				// randomly set the topics ...
				int topic = random.nextInt(numTopics);
				topics_NUMBERS[position] = topic;
//				System.out.println("tokensPerTopic_TITLE:"+tokensPerTopic_TITLE);
				tokensPerTopic_NUMBERS[topic]++;
				
				int type = tokenSequence_numbers.getIndexAtPosition(position);
				typeTopicCounts_NUMBERS[type][topic]++;
			} // 
			// KEYWORDS // randomly set the topics ...
			int[] topics_KEYWORDS = topicSequence_keywords.getFeatures();
			
			for (int position = 0; position < tokenSequence_keywords.size(); position++) {
				// randomly set the topics ...
				int topic = random.nextInt(numTopics);
				topics_KEYWORDS[position] = topic;
				tokensPerTopic_KEYWORDS[topic]++;
				
				int type = tokenSequence_keywords.getIndexAtPosition(position);
//				System.out.println("tokensPerTopic_TITLE:"+tokensPerTopic_TITLE+" numTopics:"+numTopics+" topic:"+topic+" type:"+type+" "+typeTopicCounts_KEYWORDS.length);
				typeTopicCounts_KEYWORDS[type][topic]++;
			} // 

			// KEYWORDS + MISSING related keywords
			int[] topics_KEYWORDS_N_MISSINGrelatedKEYWORDS = topicSequence_keywords_N_missingKEYWORDS.getFeatures();
			
			for (int position = 0; position < tokensSequence_keywords_N_missingKEYWORDS.size(); position++) {
				// randomly set the topics ...
				int topic = random.nextInt(numTopics);
				topics_KEYWORDS_N_MISSINGrelatedKEYWORDS[position] = topic;
				tokensPerTopic_MISSING_KEYWORDS[topic]++;
				
				int type = tokensSequence_keywords_N_missingKEYWORDS.getIndexAtPosition(position);
//				System.out.println("tokensPerTopic_TITLE:"+tokensPerTopic_TITLE+" numTopics:"+numTopics+" topic:"+topic+" type:"+type+" "+typeTopicCounts_KEYWORDS.length);
				typeTopicCounts_KEYWORDS_N_MISSINGkeywords[type][topic]++;
			} // 
			
			// FIRST LINE // randomly set the topics ...
			int[] topics_FIRST_LINE = topicSequence_fline.getFeatures();
			
			for (int position = 0; position < tokenSequence_fline.size(); position++) {
				// randomly set the topics ...
				int topic = random.nextInt(numTopics);
				topics_FIRST_LINE[position] = topic;
//				System.out.println("tokensPerTopic_TITLE:"+tokensPerTopic_TITLE);
				tokensPerTopic_FIRST_LINE[topic]++;
				
				int type = tokenSequence_fline.getIndexAtPosition(position);
				typeTopicCounts_FIRST_LINE[type][topic]++;
			} // 
			// BOOL TRADorPOLI // randomly set the topics ...
			int[] topics_BOOL_TRADorPOLI= topicSequence_bool_TRADorPOLI.getFeatures();
			
			for (int position = 0; position < tokenSequence_bool_TRADorPOLI.size(); position++) {
				// randomly set the topics ...
				int topic = random.nextInt(numTopics);
				topics_BOOL_TRADorPOLI[position] = topic;
//				System.out.println("tokensPerTopic_TITLE:"+tokensPerTopic_TITLE);
				tokensPerTopic_bool_TRADorPOLI[topic]++;
				
				int type = tokenSequence_bool_TRADorPOLI.getIndexAtPosition(position);
				typeTopicCounts_bool_TRADorPOLI[type][topic]++;
			} // 
			
			
			//debug
//			int c2=0;
//			while(c2<tokensPerTopic_TITLE.length){
//				writerDebug.append("\n tokensPerTopic_TITLE[title]:"+tokensPerTopic_TITLE[c2]+" for topic="+c2);
//				writerDebug.flush();
//				c2++;
//			}
			TopicAssignment t = new TopicAssignment (instance,
													 instance_title,
													 instance_person_location, 
													 instance_keywords, instance_fline, 
													 instance_organization, instance_numbers,
													 instance_MISSINGkeywords, instance_bool_TRADorPOLI,
													 instance_all_features,
													 topicSequence,
													 topicSequence_title,
													 topicSequence_person_location,
													 topicSequence_keywords,
													 topicSequence_fline,
													 topicSequence_organization, 
													 topicSequence_numbers,
													 topicSequence_MISSINGkeywords, 
													 topicSequence_bool_TRADorPOLI,
													 topicSequence_all_features,
													 topicSequence_keywords_N_missingKEYWORDS
													 );
 
//			TopicAssignment t = new TopicAssignment (instance, topicSequence);
			data.add (t);
		}
		}
		catch(Exception e){
			e.printStackTrace();
		}
	} //end addInstances()
	// sample()
	public void sample(int    iterations,
					   String baseFolder,
					   String inFile_authID_docIDCSV,
					   String inFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames,
					   String query_topicCSV,
					   String debugFile,
					   boolean is_SOPprint ) throws IOException {
		FileWriter writerDebug=null;
		TreeMap<Integer,TreeMap<Integer,String>> map_authID_topicIDasKEY4queryNODUP=new TreeMap<Integer,TreeMap<Integer,String>>();
		TreeMap<Integer, String> map_authID_topicIDCSVasKEY4query_withDUP=new TreeMap<Integer, String>();
		TreeMap<Integer,TreeMap<String,String>> map_authID_queryatomASkey=new TreeMap<Integer,TreeMap<String,String>>();
		TreeMap<String, Integer> map_global_queryName_queryID=new TreeMap<>();
		boolean is_YES_run_author_code=false;
		int num_of_context=7;
		try{
			//
		
			
			//create global topic ID for each query
			String [] arr_query=query_topicCSV.split(",");
			int c=0;
			// 
			while(c<arr_query.length){ //
				map_global_queryName_queryID.put(arr_query[c], c);
				c++;
			}
			// Load authID_docIDCSV
			TreeMap<Integer, String> map_eachLine_inFile_authID_docIDCSV=
			readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
			readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line( inFile_authID_docIDCSV,
																						-1, 
																						-1,
																						 "load bodytext",//debug_label,
																						 false // isPrintSOP
																						 );
			// Load auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames
			TreeMap<Integer, String> map_eachLine_inFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames=
					readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
					readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line( inFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames,
																								-1, 
																								-1,
																								 "load bodytext",//debug_label,
																								 false // isPrintSOP
																								 );
			//authid!!!docID!!!query
			for(int seq:map_eachLine_inFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames.keySet()){
				String eachLine=map_eachLine_inFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames.get(seq);
				String [] s = eachLine.split("!!!");
				int authID=Integer.valueOf(s[0]);
				String query_atom=s[2];
				int global_curr_queryID=map_global_queryName_queryID.get(query_atom);
				if(is_YES_run_author_code){ // 
					// 
					if(!map_authID_queryatomASkey.containsKey(authID)){
						TreeMap<String,String> temp=new TreeMap<>();
						temp.put(query_atom, "");
						map_authID_queryatomASkey.put(authID, temp);
						//
						TreeMap<Integer,String> temp_int=new TreeMap<>();
						temp_int.put(global_curr_queryID, "");
						map_authID_topicIDasKEY4queryNODUP.put(authID, temp_int);
						map_authID_topicIDCSVasKEY4query_withDUP.put(authID, String.valueOf(global_curr_queryID) );
					}
					else{
						TreeMap<String,String> temp=map_authID_queryatomASkey.get(authID);
						map_authID_queryatomASkey.put(authID, temp);  
						//
						TreeMap<Integer,String> temp_int=map_authID_topicIDasKEY4queryNODUP.get(authID);
						temp_int.put(global_curr_queryID, ""); // this overwrites existing
						map_authID_topicIDasKEY4queryNODUP.put(authID, temp_int);
						String new_topicID_CSV=map_authID_topicIDCSVasKEY4query_withDUP.get(authID)+","+global_curr_queryID;
						map_authID_topicIDCSVasKEY4query_withDUP.put(authID, new_topicID_CSV);
					}
				}
			}
			System.out.println("****Inside sample");
			writerDebug=new FileWriter(new File( debugFile) ,true);
			
			writerDebug.append("\n map_authID_topicIDCSVasKEY4query_withDUP:"+map_authID_topicIDCSVasKEY4query_withDUP+"\n");
			writerDebug.flush();
			
			TreeMap<Integer,Integer> map_TEMPauthID_ORIGauthID=new TreeMap<>();
			int max_AUTH_ID=-1; 
			int[][] arr_authID_TypeTopicCounts=new int[map_TEMPauthID_ORIGauthID.size()][numTopics];
			int[][] arr_context_TypeTopicCounts=new int[num_of_context][numTopics];
			//initialize
			int seq_tempAuthID=1;//(authid starts with 1)
			if(is_YES_run_author_code){
				for(int authID:map_authID_topicIDCSVasKEY4query_withDUP.keySet()){
					if(authID>max_AUTH_ID) max_AUTH_ID=authID;
					map_TEMPauthID_ORIGauthID.put(seq_tempAuthID, authID);
					seq_tempAuthID++;
				}
				// 
				for(int authID:map_authID_topicIDCSVasKEY4query_withDUP.keySet()){
					int n=0;String [] arr_topics=map_authID_topicIDCSVasKEY4query_withDUP.get(authID).split(",");
					while(n<arr_topics.length){
						arr_authID_TypeTopicCounts[authID][n]++;
						n++;
					}
				}
			}
			int c1=0; int c2=0;
			while(c1<num_of_context){ // INTITALIZE
				while(c2<numTopics){
					arr_context_TypeTopicCounts[c1][c2]=0;
					c2++;
				}
				c1++;
			}
			// for (int iteration = 1; iteration <= iterations; iteration++){ 
			for(int iteration = 1; iteration <= iterations; iteration++){
				long iterationStart = System.currentTimeMillis();
				System.out.println("****Inside sample iteration="+iteration);
				// 
				if(iteration==iterations){
					System.out.println(" calling printDocumentTopics");
					// d_by_z writing to output file
					printDocumentTopics(			new File(debugFile+"_d_z.txt"), 
													-0.0 ,  
													6 // maximum topic
										);
				}
				// BODYTEXT
				// Loop over every document in the corpus
				for (int doc = 0; doc < data.size(); doc++){
					// BODYTEXT
					FeatureSequence tokenSequence =
						(FeatureSequence) data.get(doc).instance.getData();
					LabelSequence topicSequence =
						(LabelSequence) data.get(doc).topicSequence;
					
					int total_words_in_doc=tokenSequence.size();					
					// TITLE
					FeatureSequence tokenSequence_title =
						(FeatureSequence) data.get(doc).instance_title.getData(); // inside run() -- WorkerRunnable_p6.javaLabelSequence topicSequence =
					LabelSequence topicSequence_title =
						(LabelSequence) data.get(doc).topicSequence_title; // inside run()  -- WorkerRunnable_p6.java
//					System.out.println(" topicSequence for TITLE.."+ print_LabelSequence(topicSequence_title)  );
 
					// PERSON
					FeatureSequence tokenSequence_person_location =
							(FeatureSequence) data.get(doc).instance_person_location.getData(); // inside run() -- WorkerRunnable_p6.javaLabelSequence topicSequence =
					LabelSequence topicSequence_person_location =
							(LabelSequence) data.get(doc).topicSequence_person_location; // inside run()  -- WorkerRunnable_p6.java
					
//					System.out.println("tokenSequence_person_location:"+tokenSequence_person_location);
					
					// KEYWORDS 
					FeatureSequence tokenSequence_keywords =
							(FeatureSequence) data.get(doc).instance_keywords.getData(); // inside run() -- WorkerRunnable_p6.javaLabelSequence topicSequence =
					LabelSequence topicSequence_keywords =
							(LabelSequence) data.get(doc).topicSequence_keywords; // inside run()  -- WorkerRunnable_p6.java
					// FLINE
					FeatureSequence tokenSequence_fline =
							(FeatureSequence) data.get(doc).instance_fline.getData(); // inside run() -- WorkerRunnable_p6.javaLabelSequence topicSequence =
					LabelSequence topicSequence_fline =
							(LabelSequence) data.get(doc).topicSequence_fline; // inside run()  -- WorkerRunnable_p6.java
					// ORGANIZATION
					FeatureSequence tokenSequence_organization =
							(FeatureSequence) data.get(doc).instance_organization.getData(); // inside run() -- WorkerRunnable_p6.javaLabelSequence topicSequence =
					LabelSequence topicSequence_organization =
							(LabelSequence) data.get(doc).topicSequence_organization; // inside run()  -- WorkerRunnable_p6.java					
					// LOCATION
					FeatureSequence tokenSequence_numbers =
							(FeatureSequence) data.get(doc).instance_numbers.getData(); // inside run() -- WorkerRunnable_p6.javaLabelSequence topicSequence =
					LabelSequence topicSequence_numbers =
							(LabelSequence) data.get(doc).topicSequence_numbers; // inside run()  -- WorkerRunnable_p6.java
					//MISSINGkeywords
					FeatureSequence tokenSequence_MISSINGkeywords =
							(FeatureSequence) data.get(doc).instance_MISSINGkeywords.getData(); // inside run() -- WorkerRunnable_p6.javaLabelSequence topicSequence =
					LabelSequence topicSequence_MISSINGkeywords =
							(LabelSequence) data.get(doc).topicSequence_MISSINGkeywords; // inside run()  -- WorkerRunnable_p6.java
					// bool_TRADorPOLI 
					FeatureSequence tokenSequence_bool_TRADorPOLI =
							(FeatureSequence) data.get(doc).instance_bool_TRADorPOLI.getData(); // inside run() -- WorkerRunnable_p6.javaLabelSequence topicSequence =
					LabelSequence topicSequence_bool_TRADorPOLI =
							(LabelSequence) data.get(doc).topicSequence_bool_TRADorPOLI; // inside run()  -- WorkerRunnable_p6.java
					// ALL FEATURES
					FeatureSequence tokenSequence_all_features =
							(FeatureSequence) data.get(doc).instance_all_features.getData(); // inside run() -- WorkerRunnable_p6.javaLabelSequence topicSequence =
					LabelSequence topicSequence_all_features =
							(LabelSequence) data.get(doc).topicSequence_all_features; // inside run()  -- WorkerRunnable_p6.java
					
					if(doc<3){ //debug
						
						if(iteration%100==0)
							System.out.println("<" + iteration + " out of "+iterations+"> doc:"+doc +" "
													+data.get(doc).instance.getSource() +" tokenSequence.size:"+tokenSequence.size()
													+" title:"+data.get(doc).instance.getTitle()
													 );
					}
					
//					System.out.println("-----------------"+doc+"----------------");
//					System.out.println("tokenSequence:\n"+tokenSequence );
//					System.out.println(" topicSequence:\n"+topicSequence);
//					System.out.println("---------------------------------");
		
//					System.out.println("doc:"+doc );
					// 
					TreeMap<Integer, Double> mapStat =	sampleTopicsForOneDoc(	tokenSequence, 
																				topicSequence,
																				tokenSequence_title,
																				topicSequence_title,
																				tokenSequence_all_features,
																				topicSequence_all_features,
																				tokenSequence_person_location,topicSequence_person_location,
																				tokenSequence_keywords,topicSequence_keywords,
																				tokenSequence_fline,topicSequence_fline,
																				tokenSequence_organization,topicSequence_organization,
																				tokenSequence_numbers,topicSequence_numbers,
																				tokenSequence_MISSINGkeywords,topicSequence_MISSINGkeywords,
																				tokenSequence_bool_TRADorPOLI,topicSequence_bool_TRADorPOLI,
																				iteration, 
																				iterations,
																				doc,
																				total_words_in_doc,
																				map_authID_topicIDCSVasKEY4query_withDUP,
																				map_TEMPauthID_ORIGauthID,
																				arr_authID_TypeTopicCounts,
																				max_AUTH_ID,
//																				num_of_context,
																				writerDebug,
																				is_YES_run_author_code,
																				is_SOPprint
																				);
//					System.out.println("sample iteration:"+iteration+" doc:"+doc+" debug_title_title_greater_zero:"+mapStat.get(1)
//										+" debug_title_title_lesser_zero:"+mapStat.get(2));
				}
			
	            long elapsedMillis = System.currentTimeMillis() - iterationStart;
				logger.fine(iteration + "\t" + elapsedMillis + "ms\t");

				// Occasionally print more information
				if (showTopicsInterval != 0 && iteration % showTopicsInterval == 0) {
//					if(iterations%100==0)
						logger.info("<" + iteration + " out of "+iterations+"> Log Likelihood: " 
										+ modelLogLikelihood() 
										+ "\n" 
										+ topWords (wordsPerTopic));
				}
//				System.out.println("alpha:"+alpha+" beta:"+beta+" betaSum:"+betaSum+" iteration:"+iteration);
			} //END for (int iteration = 1; iteration <= iterations; iteration++){
			System.out.println("END of sample....");
		}
		catch(Exception e){
			try{
				e.printStackTrace();
				writerDebug.append(" error in catch..."+e.getMessage()+"\n");
				writerDebug.flush();
			}
			catch(Exception e2){}
		}
	} //sample()
	
	// sampleTopicsForOneDoc
	protected TreeMap<Integer,Double> sampleTopicsForOneDoc(  FeatureSequence 	tokenSequence, //document-wise
															  FeatureSequence 	topicSequence,
															  FeatureSequence	tokenSequence_title,
															  FeatureSequence	topicSequence_title,
															  FeatureSequence 	tokenSequence_all_features,
															  FeatureSequence 	topicSequence_all_features, 
															  FeatureSequence 	tokenSequence_person_location, 
															  FeatureSequence 	topicSequence_person_location, 
															  FeatureSequence 	tokenSequence_keywords, 
															  FeatureSequence 	topicSequence_keywords, 
															  FeatureSequence 	tokenSequence_fline, 
															  FeatureSequence 	topicSequence_fline, 
															  FeatureSequence 	tokenSequence_organization, 
															  FeatureSequence 	topicSequence_organization, 
															  FeatureSequence 	tokenSequence_numbers, 
															  FeatureSequence 	topicSequence_numbers, 
															  FeatureSequence 	tokenSequence_MISSINGkeywords, 
															  FeatureSequence 	topicSequence_MISSINGkeywords, 
															  FeatureSequence 	tokenSequence_bool_TRADorPOLI, 
															  FeatureSequence 	topicSequence_bool_TRADorPOLI, 
															  int 		 		curr_iteration, 
															  int 		 		max_iterations, 
															  int				doc_index2,
															  int       		total_words_in_doc_index2,
															  TreeMap<Integer, String> map_authID_topicIDCSVasKEY4query_withDUP,
															  TreeMap<Integer,Integer> map_TEMPauthID_ORIGauthID,
															  int [][] 			arr_authID_TypeTopicCounts,
															  int				max_AUTH_ID,
															  FileWriter 		writerDebug,
															  boolean   		is_YES_run_author_code,
															  boolean			is_SOPprint ){
//		System.out.println("Inside sampleTopicsForOneDoc:");
		int entry_1=0; int entry_2=0;int entry_2_1=0;int entry_2_2=0;int entry_2_3=0;int entry_2_4=0; int entry_2_5=0;
		int entry_2_10=0;int entry_2_11=0;
		int debug_title_title_greater_zero=0;int debug_title_title_lesser_zero=0;
		TreeMap<Integer,Double> mapOut=new TreeMap<Integer,Double>();
		int curr_random_context=-1;
		boolean is_YES_debug_local_flag=false;
//		boolean is_YES_run_author_code=false;
//		int max_AUTH_ID=-1; //will be overwritten below
		try{
//		int[][] arr_authID_TypeTopicCounts=new int[10000][10000];
		// BODYTEXT
		int[] oneDocTopics = topicSequence.getFeatures();
		int[] currentTypeTopicCounts;
		int[] currentTypeTopicCounts_for_CONTEXT = null;
		int type, oldTopic, newTopic = 0;
		int type_curr_context=0;
		double topicWeightsSum;
		int docLength = tokenSequence.getLength();
		int[] localTopicCounts=new int[numTopics];
		// TITLE
		int[] oneDocTopics_TITLE = topicSequence_title.getFeatures();
		int[] currentTypeTopicCounts_TITLE;
		int type_TITLE, oldTopic_TITLE, newTopic_TITLE;
		double topicWeightsSum_TITLE; int count_debug=0; int count_debug_NEG=0;
		int docLength_TITLE = tokenSequence_title.getLength();
		int[] localTopicCounts_TITLE = new int[numTopics];
		double sample =0.; String debug_concLine="";
		//purpose to get random authID randomly..TEMPauthID are sequential, where as authID in map_authID_topicIDCSVasKEY4query_withDUP NOT SEQEUNTIAL..
//		TreeMap<Integer,Integer> map_TEMPauthID_ORIGauthID=new TreeMap<>();
		
		// populate topic counts (BODYTEXT)
		for(int position = 0; position < docLength; position++){
			localTopicCounts[oneDocTopics[position]]++;
		}
		// initialize
//		for(int topic = 0; topic < numTopics; topic++){
//			localTopicCounts_TITLE[topic]=1;
//		}
		//		populate topic counts (TITLE)
//		for (int position1 = 0; position1 < docLength_TITLE; position1++) {
////			System.out.println("SETTING TITLE ");
//			localTopicCounts_TITLE[oneDocTopics_TITLE[position1]]++;
//		}
		//BODYTEXT
		double score = 0, sum;
		double[] topicTermScores = new double[numTopics];
		//TITLE
		double score_TITLE, sum_TITLE;
		double[] topicTermScores_TITLE = new double[numTopics];
		double sum_localTopicCounts_for_CONTEXT=0.;

//			System.out.println("1:"+typeTopicCounts_TITLE.length+" 2:"+typeTopicCounts_PERSON_LOCATION.length
//					+" 3:"+typeTopicCounts_KEYWORDS.length+" 4:"+typeTopicCounts_FIRST_LINE.length
//					+" 5:"+typeTopicCounts_ORGANIZATION.length //+" 6:"+typeTopicCounts_NUMBERS[0].length +" 7:"+typeTopicCounts_MISSING_KEYWORDS[0].length
//					);
		
		
		int length_of_ALL_TYPES_typeTopicCounts_curr_CONTEXT=typeTopicCounts_TITLE[0].length+typeTopicCounts_PERSON_LOCATION[0].length
									+typeTopicCounts_KEYWORDS[0].length+typeTopicCounts_FIRST_LINE[0].length
 									+typeTopicCounts_ORGANIZATION[0].length ;//  + typeTopicCounts_MISSING_KEYWORDS[0].length;
		int dict_length_curr_CONTEXT=0;
		 
		//	Iterate over the positions (words) in the document  (-----------bodyTEXT)
		for(int position = 0; position < docLength; position++){
			int c10=0;
			FeatureSequence temp_curr_context_tokenSequence=null;
			//get NON-ZERO tokenSequence
			while(c10<11){
				curr_random_context=random.nextInt(num_of_context);

				//////////////////
				if(curr_random_context==0){
					temp_curr_context_tokenSequence=tokenSequence_title;
					typeTopicCounts_curr_CONTEXT=typeTopicCounts_TITLE;
					dict_length_curr_CONTEXT=dict_len_TITLE;
				}
				if(curr_random_context==1){
					temp_curr_context_tokenSequence=tokenSequence_person_location;
					typeTopicCounts_curr_CONTEXT=typeTopicCounts_PERSON_LOCATION;
					dict_length_curr_CONTEXT=dict_len_PERSON_LOCATION;
				}
				if(curr_random_context==2){
					temp_curr_context_tokenSequence=tokenSequence_MISSINGkeywords;  //this includes KEYWORDS+missing keywords
					typeTopicCounts_curr_CONTEXT=typeTopicCounts_KEYWORDS_N_MISSINGkeywords;
					dict_length_curr_CONTEXT=dict_len_KEYWORDS_missingKEYWORDS;
				}
				if(curr_random_context==3){
					temp_curr_context_tokenSequence=tokenSequence_fline;
					typeTopicCounts_curr_CONTEXT=typeTopicCounts_FIRST_LINE;
					dict_length_curr_CONTEXT=dict_len_FIRST_LINE;
				}
				if(curr_random_context==4){ 
					temp_curr_context_tokenSequence=tokenSequence_organization;
					typeTopicCounts_curr_CONTEXT=typeTopicCounts_ORGANIZATION;
					dict_length_curr_CONTEXT=dict_len_ORGANIZATION;
				}
				if(curr_random_context==5){ // location
					temp_curr_context_tokenSequence=tokenSequence_numbers;
					typeTopicCounts_curr_CONTEXT= typeTopicCounts_NUMBERS;// typeTopicCounts_NUMBERS;
					dict_length_curr_CONTEXT=dict_len_NUMBERS;
				}
				///ENDS.....
//				if(curr_random_context==6){
//					temp_curr_context_tokenSequence=tokenSequence_MISSINGkeywords;
//					typeTopicCounts_curr_CONTEXT=typeTopicCounts_MISSING_KEYWORDS;
//				}
//				if(curr_random_context==7){
//					temp_curr_context_tokenSequence=tokenSequence_bool_TRADorPOLI;
//					typeTopicCounts_curr_CONTEXT=typeTopicCounts_bool_TRADorPOLI;
//				}
//				if(curr_random_context==8){
//					temp_curr_context_tokenSequence=tokenSequence_all_features;
////					typeTopicCounts_curr_CONTEXT=typeTopicCounts_;
//				}
				
//				length_of_ALL_TYPES_typeTopicCounts_curr_CONTEXT=typeTopicCounts_curr_CONTEXT[0].length;
				if(temp_curr_context_tokenSequence.getLength()>0 && typeTopicCounts_curr_CONTEXT!=null ) break;
				c10++;
			}
			/// SAMPLE a word from CONTEXT
			int type_from_random_word_from_temp_curr_context_tokenSequence = -1;int random_word_from_temp_curr_context_tokenSequence=-1;
			int len=temp_curr_context_tokenSequence.getLength();
			try{
//				writerDebug.append("temp_curr_context_tokenSequence.leng:"+len+" curr_random_context:"+curr_random_context+"\n");
				random_word_from_temp_curr_context_tokenSequence=random.nextInt(len);
//				writerDebug.append(" random_word_from_temp_curr_context_tokenSequence:"+random_word_from_temp_curr_context_tokenSequence
//									+" temp_curr_context_tokenSequence.leng:"+len+" curr_random_context:"+curr_random_context+"\n"
//								   );
//				writerDebug.append("--------------------\n");
				writerDebug.flush();
			}
			catch(Exception e){
				writerDebug.append("\n CATCH ERROR:"+e.getMessage()
								+" random_word_from_temp_curr_context_tokenSequence:"+random_word_from_temp_curr_context_tokenSequence
							 	+" temp_curr_context_tokenSequence.leng:"+len+" curr_random_context:"+curr_random_context);
				writerDebug.flush();
			}
			// (type) CONTEXT 
			type_from_random_word_from_temp_curr_context_tokenSequence =
											temp_curr_context_tokenSequence.getIndexAtPosition(random_word_from_temp_curr_context_tokenSequence);
			
//			System.out.println("token:"+type_from_random_word_from_temp_curr_context_tokenSequence+" curr_random_context:"+curr_random_context);	
			
			// (type) base version - body Text 
			type = tokenSequence.getIndexAtPosition(position);
			oldTopic = oneDocTopics[position];
			// Grab the relevant row from our two-dimensional array
			currentTypeTopicCounts=typeTopicCounts[type];
			//	Remove this token from all counts.
//			typeTopicCounts_curr_CONTEXT[type_from_random_word_from_temp_curr_context_tokenSequence][oldTopic]--; // equ.2
			currentTypeTopicCounts_for_CONTEXT=typeTopicCounts_curr_CONTEXT[type_from_random_word_from_temp_curr_context_tokenSequence]; //equ5 (Comment/uncomment)
			// 
			localTopicCounts[oldTopic]--; //comment/uncomment
			tokensPerTopic[oldTopic]--; //comment/uncomment
//			if(currentTypeTopicCounts_for_CONTEXT[oldTopic]>0)
//				currentTypeTopicCounts_for_CONTEXT[oldTopic]--;
			assert(tokensPerTopic[oldTopic] >= 0) : "old Topic " + oldTopic + " below 0";
			currentTypeTopicCounts[oldTopic]--;
			
			if(curr_random_context==0){ //TITLE
				tokensPerTopic_TITLE[oldTopic]--;
			}if(curr_random_context==1){ //_PERSON;
				tokensPerTopic_PERSON_LOCATION[oldTopic]--;
			}if(curr_random_context==2){//this includes KEYWORDS+missing keywords
				tokensPerTopic_MISSING_KEYWORDS[oldTopic]--;
			}if(curr_random_context==3){ //_FIRST_LINE;
				tokensPerTopic_FIRST_LINE[oldTopic]--;
			}if(curr_random_context==4){  //_ORGANIZATION;
				tokensPerTopic_ORGANIZATION[oldTopic]--;
			}if(curr_random_context==5){ // _LOCATION
				tokensPerTopic_NUMBERS[oldTopic]--;
			}
			
			
			int curr_rand_TEMPauthID=-1;int curr_rand_authID=-1;
			int do_10_random_till_valid_authId=0; //some authID missing as they are dirty and hence removed in between sequences of continous auth id
			
			// Now calculate and add up the scores for each topic for this word
			sum = 0.0;
			double sum_localTopicCounts=0.;double sum_localTopicCounts_TITLE=0.;
			double sum_localTopicCounts_for_AUTHOR=0.;
			
			double sum_localTopicCounts_for_CONTEXT_only_Topic=0.;
			debug_concLine="";
				//System.out.println("entry 4.44");
			// 
			for(int topic = 0; topic < numTopics; topic++){
				sum_localTopicCounts+=localTopicCounts[topic];
			}
			//
			for(int titleID = 0; titleID < numTopics; titleID++){
				sum_localTopicCounts_TITLE+=localTopicCounts_TITLE[titleID];
			}
			String debug_sum_localTopicCounts_for_CONTEXT="";String debug_sum_localTopicCounts_for_CONTEXT2="";

			// CONTEXT
			for(int topic = 0; topic < numTopics; topic++){
//				System.out.println("3:"+typeTopicCounts_curr_CONTEXT[type_from_random_word_from_temp_curr_context_tokenSequence][topic]);
				sum_localTopicCounts_for_CONTEXT+= typeTopicCounts_curr_CONTEXT[type_from_random_word_from_temp_curr_context_tokenSequence][topic]; //equ2
//				sum_localTopicCounts_for_CONTEXT=sum_localTopicCounts_for_CONTEXT+currentTypeTopicCounts_for_CONTEXT[topic]; //equ5
				if(debug_sum_localTopicCounts_for_CONTEXT.length()==0){
					debug_sum_localTopicCounts_for_CONTEXT=
															String.valueOf(currentTypeTopicCounts_for_CONTEXT[topic]);
//															String.valueOf(typeTopicCounts_curr_CONTEXT[type_from_random_word_from_temp_curr_context_tokenSequence][topic]);
					debug_sum_localTopicCounts_for_CONTEXT2=String.valueOf(sum_localTopicCounts_for_CONTEXT);
				}
				else{
					debug_sum_localTopicCounts_for_CONTEXT= debug_sum_localTopicCounts_for_CONTEXT+"```"+
															String.valueOf(currentTypeTopicCounts_for_CONTEXT[topic]);
//															String.valueOf(typeTopicCounts_curr_CONTEXT[type_from_random_word_from_temp_curr_context_tokenSequence][topic]);
					debug_sum_localTopicCounts_for_CONTEXT2=debug_sum_localTopicCounts_for_CONTEXT2+"~~~"+String.valueOf(sum_localTopicCounts_for_CONTEXT);
				}
			} // end CONTEXT 
			//CONTEXT (only TOPIC)
			for(int topic2 = 0; topic2 < numTopics; topic2++){
				sum_localTopicCounts_for_CONTEXT_only_Topic+=typeTopicCounts_curr_CONTEXT_onlyTopic[topic2];
			}
			
			//is_YES_run_author_code
			if(is_YES_run_author_code){
				String [] arr_topics=map_authID_topicIDCSVasKEY4query_withDUP.get(curr_rand_authID).split(",");
				int c=0;
				while(c<arr_topics.length){
					sum_localTopicCounts_for_AUTHOR+=arr_authID_TypeTopicCounts[curr_rand_authID][c];
					c++;
				}
			}
			//System.out.println("entry 11");
			// Here's where the math happens! Note that overall performance is 
			//  dominated by what you do in this loop.
			for(int topic = 0; topic < numTopics; topic++){
				// lenin: refer http://www.jmlr.org/proceedings/papers/v13/xiao10a/xiao10a.pdf
 
								 
				if(sum_localTopicCounts_for_CONTEXT<=0){
//					writerDebug.append("\n negative 2:"+sum_localTopicCounts_for_CONTEXT+" doc_index2:"+doc_index2+"--"+debug_sum_localTopicCounts_for_CONTEXT
//														+" top:"+topic+"<--->"+debug_sum_localTopicCounts_for_CONTEXT2);
					sum_localTopicCounts_for_CONTEXT=1;entry_2_11++;
				}
				writerDebug.flush();
//				if(typeTopicCounts_curr_CONTEXT[type_from_random_word_from_temp_curr_context_tokenSequence][topic] >= 0.&& sum_localTopicCounts_for_CONTEXT > 0.){ //for equ1
//				if(currentTypeTopicCounts_for_CONTEXT[topic]>=0 && sum_localTopicCounts_for_CONTEXT>0.){ // for equ5
					
					
//					score =  (gamma + currentTypeTopicCounts_for_CONTEXT[topic] )/
//											((dict_length_curr_CONTEXT*gamma) +tokensPerTopic_CONTEXT[curr_random_context][topic]) *
//							 ((beta + currentTypeTopicCounts[topic]) / (betaSum + tokensPerTopic[topic]))*
//							 (alpha + localTopicCounts[topic])/( numTopics * alpha + sum_localTopicCounts ) ;  // NOT COLLAPSED equ6<--best(implemented global words for each context)

					
					score =  (gamma + currentTypeTopicCounts_for_CONTEXT[topic] )/
									((dict_length_curr_CONTEXT*gamma) +tokensPerTopic_CONTEXT[curr_random_context][topic]) *
								((beta + currentTypeTopicCounts[topic]) / (betaSum + tokensPerTopic[topic]))*
								(alpha + localTopicCounts[topic])  ;  //  COLLAPSED equ6
					
 
					entry_1++;
//				}
				
				 
				//
				sum += score;
				topicTermScores[topic]=score;
				//System.out.println("entry 5");
				if(is_YES_debug_local_flag){
					if(debug_concLine.length()==0){
						debug_concLine="\n\nscore:"+score+" 1:"+localTopicCounts[topic]+" 2:"+currentTypeTopicCounts[topic]
												+" 3:"+tokensPerTopic[topic] +" 4:"+localTopicCounts_TITLE[topic]
												+" 5:"+sum_localTopicCounts_TITLE
												+" 6:"+arr_authID_TypeTopicCounts[curr_rand_authID][topic]
//												+" topic:"+topic +" d1:"+d1+" d2:"+d2+" d3:"+d3 +" d1*d2*d3:"+(d1*d2*d3)
												+" sum:"+sum+" score:"+score+" topic:"+topic+"\n";
					}
					else{
						debug_concLine=debug_concLine+"<-->"+"score:"+score+" 1:"+localTopicCounts[topic]+" 2:"+currentTypeTopicCounts[topic]
												+" 3:"+tokensPerTopic[topic]+" 4:"+localTopicCounts_TITLE[topic]
												+" 5:"+sum_localTopicCounts_TITLE
												+" 6:"+arr_authID_TypeTopicCounts[curr_rand_authID][topic]
//												+" topic:"+topic+" d1:"+d1+" d2:"+d2+" d3:"+d3 +" d1*d2*d3:"+(d1*d2*d3)
												+" sum:"+sum+" score:"+score+" topic:"+topic+"\n";
					}
				}
				//System.out.println("entry 6");
				// 
				if(is_SOPprint){
					System.out.println("doc_index2:"+doc_index2+ " out of total="+total_words_in_doc_index2
										+ " topicTermScore:"+ topicTermScores[topic]+" "+topic+ " "+score);
					writerDebug.append("doc_index2:"+doc_index2+ " out of total="+total_words_in_doc_index2
										+ " topicTermScore:"+ topicTermScores[topic]+" "+topic+ " "+score+"\n");
					writerDebug.flush();
				}
				// 
				if(is_YES_debug_local_flag){
					if((topic==numTopics-1 && sum<0. ) ||(topic==numTopics-1)){// && d1<0. ) ){
						System.out.println(" NEG ERROR: typeTopicCounts[type]:"+typeTopicCounts[type]
											+" localTopicCounts[topic]:"+localTopicCounts[topic]
											+" tokensPerTopic[topic]:"+tokensPerTopic[topic]+" score:"+score+"\n");
						count_debug_NEG++;
						if(count_debug_NEG<=2)
							writerDebug.append(  "(NEG) typeTopicCounts[type]:"+typeTopicCounts[type]
												+" localTopicCounts[topic]:"+localTopicCounts[topic]
												+" tokensPerTopic[topic]:"+tokensPerTopic[topic]
												+" sum_localTopicCounts_TITLE:"+sum_localTopicCounts_TITLE
												+" (gamma + localTopicCounts_TITLE[topic] ):"+(gamma + localTopicCounts_TITLE[topic] )
												+" (betaSum + tokensPerTopic[topic]):"+(betaSum + tokensPerTopic[topic])
												+" localTopicCounts_TITLE[topic]:"+localTopicCounts_TITLE[topic]
												+" arr_authID_TypeTopicCounts[curr_rand_authID][topic]:"+arr_authID_TypeTopicCounts[curr_rand_authID][topic]
												+" sum_localTopicCounts_for_AUTHOR:"+sum_localTopicCounts_for_AUTHOR
												+" debug_concLine:"+debug_concLine
												+"\n newTopic:"+newTopic
												+"\n");
							writerDebug.flush();
					}
					else if(topic==numTopics-1 ){
						count_debug++;
						if(count_debug<=10){
							writerDebug.append("(POSITIVE) typeTopicCounts[type]:"+typeTopicCounts[type]
												+" localTopicCounts[topic]:"+localTopicCounts[topic]
												+" tokensPerTopic[topic]:"+tokensPerTopic[topic]
												+" sum_localTopicCounts_TITLE:"+sum_localTopicCounts_TITLE
												+" sum:"+sum
												+" score:"+score +" topic:"+topic
												+" localTopicCounts_TITLE[topic]:"+localTopicCounts_TITLE[topic]
												+"\n");
							writerDebug.flush();
						}
					}
				}
			} // for(int topic = 0; topic < numTopics; topic++){
			//System.out.println("entry 7");
			double uniform=random.nextUniform();
			// Choose a random point between 0 and the sum of all topic scores
			sample = uniform * sum;
			// Figure out which topic contains that point
			newTopic = -1;
			// debug 
//			if(sample <=0. || curr_iteration ==1){ // 
//				writerDebug.append("\n BFR sample(bodyTEXT):"+sample+ " uniform:"+uniform+" sum:"+sum+
//								" position:"+position+" newTopic:"+newTopic+" curr_iteration:"+curr_iteration
//								+" sample:"+(sample > 0.0));
//				writerDebug.flush();
//		    }
//			System.out.println("BFR sample:"+sample+ " position:"+position+" newTopic:"+newTopic);
			//System.out.println("entry 8");
			while(sample > 0.0){
				newTopic++;
				sample -= topicTermScores[newTopic];
			}
//			System.out.println("AFT sample:"+sample+ " position:"+position+" newTopic:"+newTopic);
			// Make sure we actually sampled a topic
			if(newTopic == -1) {
//				writerDebug.append("\n error: typeTopicCounts[type]:"+typeTopicCounts[type] +" sample:"+sample+ " uniform:"+uniform+ " sum:"+sum+"\n");writerDebug.flush();
//				continue; // lenin added
//				throw new IllegalStateException ("SimpleLDA: New topic not sampled."); // lenin commented
			}
			// Put that new topic into the counts
			oneDocTopics[position] = newTopic;
			localTopicCounts[newTopic]++;
			tokensPerTopic[newTopic]++;
			currentTypeTopicCounts[newTopic]++;
//			typeTopicCounts_curr_CONTEXT[type_from_random_word_from_temp_curr_context_tokenSequence][newTopic]++;
			
			if(curr_random_context==0){ //TITLE
				tokensPerTopic_TITLE[newTopic]++;
			}if(curr_random_context==1){ //_PERSON
				tokensPerTopic_PERSON_LOCATION[newTopic]++;
			}if(curr_random_context==2){//this includes KEYWORDS+missing keywords
				tokensPerTopic_MISSING_KEYWORDS[newTopic]++;
			}if(curr_random_context==3){ //_FIRST_LINE;
				tokensPerTopic_FIRST_LINE[newTopic]++;
			}if(curr_random_context==4){  //_ORGANIZATION;
				tokensPerTopic_ORGANIZATION[newTopic]++;
			}if(curr_random_context==5){ // _LOCATION
				tokensPerTopic_NUMBERS[newTopic]++;
			}
						
			//TITLE
//			localTopicCounts_TITLE[newTopic]++;
//			tokensPerTopic_TITLE[newTopic]++;
//			currentTypeTopicCounts_TITLE[newTopic]++;
			//
			if(is_YES_run_author_code)
				arr_authID_TypeTopicCounts[curr_rand_authID][newTopic]++;
		} // for(int position = 0; position < docLength; position++) {

		
		writerDebug.append("\n entry_1:"+entry_1+" entry_2:"+entry_2+" 1:"+entry_2_1+" 2:"+entry_2_2+" 3:"+entry_2_3+" 4:"+entry_2_4+" 5:"+entry_2_5
								+" 6:"+entry_2_10+" 7:"+entry_2_11);
		writerDebug.flush();
		
		// --------- LENIN ADDED FOR d by z probability --------------
		int num_of_topics=10; int doc_index=0;
		Formatter out=null;		
		if(curr_iteration==max_iterations){
			// 
	        while(doc_index < 20000){
	    		String conc_doc_prob="";
		        for (int topic = 0; topic < num_of_topics; topic++) {
		        		out = new Formatter(new StringBuilder(), Locale.US);
//			            out.format("print1: %d\t%.3f\t", topic, model.getTopicProbabilities(doc_index)[topic]);
		        		if(conc_doc_prob.length() ==0){
		        			conc_doc_prob=topic + ":"+ oneDocTopics[topic];
		        			 
		        		}
		        		else{
	//	        			conc_doc_prob=conc_doc_prob+" "+topic + ":"+  oneDocTopics[topic];
		        		}
			            int[][]s=getTypeTopicCounts();
			            if(is_SOPprint){
				            if(doc_index<=3){
					            System.out.println("doc:"+doc_index+" "+ s[doc_index][topic]+"\n");
					            writerDebug.append("doc:"+doc_index+" "+ s[doc_index][topic]+"\n");
					            writerDebug.flush();
				            }
			            }
		        }
	            doc_index++;
	//        	System.out.println("doc:"+doc_index+" "+conc_doc_prob);    
	        }
		} // if(curr_iteration==1){
	
		mapOut.put(1, Double.valueOf(debug_title_title_greater_zero));
		mapOut.put(2, Double.valueOf(debug_title_title_lesser_zero));
		}
		catch(Exception e){
			e.printStackTrace();
		}
		return mapOut;
	}
	//
	public double modelLogLikelihood() {
		System.out.println("****inside modelLogLikelihood");
		double logLikelihood = 0.0;

		// The likelihood of the model is a combination of a 
		// Dirichlet-multinomial for the words in each topic
		// and a Dirichlet-multinomial for the topics in each
		// document.

		// The likelihood function of a dirichlet multinomial is
		//	 Gamma( sum_i alpha_i )	 prod_i Gamma( alpha_i + N_i )
		//	prod_i Gamma( alpha_i )	  Gamma( sum_i (alpha_i + N_i) )

		// So the log likelihood is 
		//	logGamma ( sum_i alpha_i ) - logGamma ( sum_i (alpha_i + N_i) ) + 
		//	 sum_i [ logGamma( alpha_i + N_i) - logGamma( alpha_i ) ]

		// Do the documents first

		int[] topicCounts = new int[numTopics];
		double[] topicLogGammas = new double[numTopics];
		int[] docTopics;

		for (int topic=0; topic < numTopics; topic++) {
			topicLogGammas[ topic ] = Dirichlet.logGamma( alpha );
		}
		// 
		for (int doc=0; doc < data.size(); doc++) {
			LabelSequence topicSequence = (LabelSequence) data.get(doc).topicSequence;
			docTopics = topicSequence.getFeatures();

			for (int token=0; token < docTopics.length; token++) {
//				System.out.println("docTopics.length:"+docTopics.length+" "+docTopics[doc] );
				topicCounts[ docTopics[token] ]++;
			}

			for (int topic=0; topic < numTopics; topic++) {
				if (topicCounts[topic] > 0) {
					logLikelihood += (Dirichlet.logGamma(alpha + topicCounts[topic]) -
									  topicLogGammas[ topic ]);
				}
			}

			// subtract the (count + parameter) sum term
			logLikelihood -= Dirichlet.logGamma(alphaSum + docTopics.length);

			Arrays.fill(topicCounts, 0);
		}
	
		// add the parameter sum term
		logLikelihood += data.size() * Dirichlet.logGamma(alphaSum);

		// And the topics
		double logGammaBeta = Dirichlet.logGamma(beta);
		for (int type=0; type < numTypes; type++) {
			// reuse this array as a pointer

			topicCounts = typeTopicCounts[type];

			for (int topic = 0; topic < numTopics; topic++) {
				if (topicCounts[topic] == 0) { continue; }
				
				logLikelihood += Dirichlet.logGamma(beta + topicCounts[topic]) - logGammaBeta;

				if (Double.isNaN(logLikelihood)) {
					System.out.println("Inside modelLogLikelihood....Exiting ..."+" gamma:"+gamma+" beta:"+beta);
					System.out.println(topicCounts[topic]);
					System.exit(1);
				}
			}
		}
		// 
		for (int topic=0; topic < numTopics; topic++){
			logLikelihood -= Dirichlet.logGamma( (beta * numTypes) + tokensPerTopic[ topic ] );
			if (Double.isNaN(logLikelihood)) {
				System.out.println("Exiting.....after topic " + topic + " " + tokensPerTopic[ topic ]+" gamma:"+gamma+" beta:"+beta);
				System.exit(1);
			}
		}
	
		logLikelihood += numTopics * Dirichlet.logGamma(beta * numTypes);

		if (Double.isNaN(logLikelihood)) {
			System.out.println("at the end");
			System.exit(1);
		}


		return logLikelihood;
	}

	// 
	// Methods for displaying and saving results
	//

	public String topWords(int numWords) {

		StringBuilder output = new StringBuilder();

		IDSorter[] sortedWords = new IDSorter[numTypes];

		for (int topic = 0; topic < numTopics; topic++) {
			for (int type = 0; type < numTypes; type++) {
				sortedWords[type] = new IDSorter(type, typeTopicCounts[type][topic]);
			}

			Arrays.sort(sortedWords);
			
			output.append(topic + "\t" + tokensPerTopic[topic] + "\t");
			for (int i=0; i < numWords; i++) {
				output.append(alphabet.lookupObject(sortedWords[i].getID()) + " ");
			}
			output.append("\n");
		}

		return output.toString();
	}

	 // another source: http://www.programcreek.com/java-api-examples/index.php?api=cc.mallet.types.IDSorter
	/**
	 *  @param file        The filename to print to
	 *  @param threshold   Only print topics with proportion greater than this number
	 *  @param max         Print no more than this many topics
	 */
	public void printDocumentTopics(File file, double threshold, int max) throws IOException {
		System.out.println("Inside printDocumentTopics():"+data.size());
		PrintWriter out = new PrintWriter(file);

		out.print ("#doc source topic proportion ...\n");
		int docLen;
		int[] topicCounts = new int[ numTopics ];
		 
		IDSorter[] sortedTopics = new IDSorter[ numTopics ];
		for (int topic = 0; topic < numTopics; topic++) {
			// Initialize the sorters with dummy values
			sortedTopics[topic] = new IDSorter(topic, topic);
		}
//		System.out.println("entry 2");
		if (max < 0 || max > numTopics) {
			max = numTopics;
		}
		 
		// 
		for (int doc = 0; doc < data.size(); doc++){
//			System.out.println("doc:"+doc);
			LabelSequence topicSequence = (LabelSequence) data.get(doc).topicSequence;
			int[] currentDocTopics = topicSequence.getFeatures();

			out.print (doc); out.print (' ');
			//
			if (data.get(doc).instance.getSource() != null) {
				out.print (data.get(doc).instance.getSource()); 
			}
			else {
				 //lenin: getTarget will print the label from input file
				//out.print (data.get(doc).instance.getName()+" "+data.get(doc).instance.getTarget() );
				out.print (data.get(doc).instance.getName());
			}
			out.print (' ');
			docLen = currentDocTopics.length;
			// Count up the tokens
			for (int token=0; token < docLen; token++) {
				topicCounts[ currentDocTopics[token] ]++;
			}

			// And normalize
			for (int topic = 0; topic < numTopics; topic++) {
				sortedTopics[topic].set(topic, (float) topicCounts[topic] / docLen);
			}
			
			Arrays.sort(sortedTopics);
			
			for (int i = 0; i < max; i++) {
//				if (sortedTopics[i].getWeight() < threshold) { break; }
				
				out.print (sortedTopics[i].getID() + " " + 
						  sortedTopics[i].getWeight() + " ");
			}
			out.print (" \n");

			Arrays.fill(topicCounts, 0);
		}
//		System.out.println("entry 4");
	}
	
	public void printState (File f) throws IOException {
		PrintStream out =
			new PrintStream(new GZIPOutputStream(new BufferedOutputStream(new FileOutputStream(f))));
		printState(out);
		out.close();
	}
	
	public void printState (PrintStream out) {

		out.println ("#doc source pos typeindex type topic");

		for (int doc = 0; doc < data.size(); doc++) {
			FeatureSequence tokenSequence =	(FeatureSequence) data.get(doc).instance.getData();
			LabelSequence topicSequence =	(LabelSequence) data.get(doc).topicSequence;

			String source = "NA";
			if (data.get(doc).instance.getSource() != null) {
				source = data.get(doc).instance.getSource().toString();
			}

			for (int position = 0; position < topicSequence.getLength(); position++) {
				int type = tokenSequence.getIndexAtPosition(position);
				int topic = topicSequence.getIndexAtPosition(position);
				out.print(doc); out.print(' ');
				out.print(source); out.print(' '); 
				out.print(position); out.print(' ');
				out.print(type); out.print(' ');
				out.print(alphabet.lookupObject(type)); out.print(' ');
				out.print(topic); out.println();
			}
		}
	}
	
	
	// Serialization
	
	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;
	private static final int NULL_INTEGER = -1;
	
	public void write (File f) {
		try {
			System.out.println("****Inside write() ");
			ObjectOutputStream oos = new ObjectOutputStream (new FileOutputStream(f));
			oos.writeObject(this);
			oos.close();
		}
		catch (IOException e) {
			System.err.println("Exception writing file " + f + ": " + e);
		}
	}
	
	private void writeObject(ObjectOutputStream out) throws IOException {
		out.writeInt (CURRENT_SERIAL_VERSION);

		// Instance lists
		out.writeObject (data);
		out.writeObject (alphabet);
		out.writeObject (topicAlphabet);

		out.writeInt (numTopics);
		out.writeObject (alpha);
		out.writeDouble (beta);
		out.writeDouble (betaSum);

		out.writeInt(showTopicsInterval);
		out.writeInt(wordsPerTopic);

		out.writeObject(random);
		out.writeObject(formatter);
		out.writeBoolean(printLogLikelihood);

		out.writeObject (typeTopicCounts);

		for (int ti = 0; ti < numTopics; ti++) {
			out.writeInt (tokensPerTopic[ti]);
		}
	}
	
	private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
		System.out.println("****Inside readObject() data, alphabet, topicAlphabet, numTopics, showTopicsInterval, showTopicsInterval, wordsPerTopic, typeTopicCounts, tokensPerTopic");
		int featuresLength;
		int version = in.readInt ();

		data = (ArrayList<TopicAssignment>) in.readObject ();
		alphabet = (Alphabet) in.readObject();
		topicAlphabet = (LabelAlphabet) in.readObject();

		numTopics = in.readInt();
		alpha = in.readDouble();
		alphaSum = alpha * numTopics;
		beta = in.readDouble();
		betaSum = in.readDouble();

		showTopicsInterval = in.readInt();
		wordsPerTopic = in.readInt();

		random = (Randoms) in.readObject();
		formatter = (NumberFormat) in.readObject();
		printLogLikelihood = in.readBoolean();
		
		int numDocs = data.size();
		this.numTypes = alphabet.size();

		typeTopicCounts = (int[][]) in.readObject();
		tokensPerTopic = new int[numTopics];
		for (int ti = 0; ti < numTopics; ti++) {
			tokensPerTopic[ti] = in.readInt();
		}
		// TITLE -- below LENIN added
//		tokensPerTopic_TITLE = new int[numTopics];
//		for (int ti = 0; ti < numTopics; ti++) {
//			tokensPerTopic_TITLE[ti] = in.readInt();
//		}
	}

	//
	public static void main(String[] args) throws IOException {
		
		try{
			//PRE-REQUISTION
			// (1) Change variables "max_AUTH_ID", "num_of_topics", "flag_loading_instances", "query_topicCSV", "num_of_iterations"
			
			long t0 = System.nanoTime();
			FileWriter writer=null;
	        FileWriter writer2_d_z_prob=null; FileWriter writer2_d_z_prob_TITLE=null;
	        FileWriter writer2_d_z_prob_AVG=null; 
			int num_of_iterations=50;
			int num_of_documents=0;
			FileWriter writerDebug=null;
//		 	Class.forName("InstanceList");

			TreeMap<Integer, String> map_lineNo_line=new TreeMap<Integer, String>();
		    String baseFolder="/Users/lenin/Dropbox/#problems/p6/merged_used_for_INTELLIGENCE_course_7xxx/dummy.mergeall/ds10/";
			String outFile=baseFolder+"LDA_d_by_z.txt"; //OUTPUT DEFAULT ="LDA_d_by_z.txt"
			String outFile2_d_by_z_prob=baseFolder+"LDA_d_by_z_prob.txt";  //OUTPUT DEFAULT="LDA_d_by_z_prob.txt"
			String outFile2_d_by_z_prob_TITLE=baseFolder+"LDA_d_by_z_prob_TITLE.txt";
			String outFile2_d_by_z_prob_AVG=baseFolder+"LDA_d_by_z_prob_AVG.txt";
			//obsolete
			String query_topicCSV="syria,boko haram,climate change,election,india,crime";
			
			writer=new FileWriter(new File(outFile));
	        writer2_d_z_prob=new FileWriter(new File(outFile2_d_by_z_prob));
	        writer2_d_z_prob_TITLE=new FileWriter(new File(outFile2_d_by_z_prob_TITLE));
	        writer2_d_z_prob_AVG=new FileWriter(new File(outFile2_d_by_z_prob_AVG));
			String 	debugFile=baseFolder+"debug__.txt";
		    int 	num_of_topics=6;
		    String  delimiter_of_inFile="!#!#";
		    String  inFile_stopwords="/Users/lenin/Dropbox/workspace-sts-3.4.0.RELEASE.macmini/grmm.malletBak/stopwords/en.txt";
	        String  inFile= "";
	        	    inFile=baseFolder+"mergedall-2016-T9-10-Topics-all-tokens.txt_ONLY_ENGLISH.txt_removed_NOISE.txt_REMOVE_STOPWORDS_FINAL.txt_4_mallet_dummy.txt";
	        	    inFile=baseFolder+"mergedall-2016-T9-10-Topics-all-tokens.txt_ONLY_ENGLISH.txt_removed_NOISE.txt_REMOVE_STOPWORDS.txt_MATCHED.txt_4_mallet.txt"; //this doesnt have 12th token
	        	    inFile=baseFolder+"mergedall-2016-T9-10-Topics-all-tokens.txt_ONLY_ENGLISH.txt_removed_NOISE.txt_REMOVE_STOPWORDS.txt_MATCHED.txt_4_mallet.txt";//12th token included
	        
	   		writerDebug=new FileWriter(new File(debugFile));
	        int flag_loading_instances=3; //tested on value=3 for p8 co-ranking problem,  {1,2}
	        
	        boolean run_including_LDA=false;
	        	   
        	   // InstanceList training = InstanceList.load (new File(inFile));
               // Begin by importing documents from text to feature sequences
	            ArrayList<Pipe> pipeList = new ArrayList<Pipe>();
	            InstanceList instances = null;
	            InstanceList instances_bodyText =null;
	    		InstanceList instances_title =null;
	    		InstanceList instances_keywords =null;
	    		InstanceList instances_MISSINGkeywords =null;
	    		InstanceList instances_person_location =null;
	    		InstanceList instances_bool_TRADorPOLI =null;
	    		InstanceList instances_fline =null;
	    		InstanceList instances_organization =null;
	    		InstanceList instances_numbers =null;
	    		InstanceList instances_all_features =null;
	    		InstanceList instances_keywords_N__MISSINGkeywords=null;
	            		
	            if(flag_loading_instances==1){
		            // Pipes: lowercase, tokenize, remove stopwords, map to features
		            pipeList.add( new CharSequenceLowercase() );
		            pipeList.add( new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")) );
		            pipeList.add( new TokenSequenceRemoveStopwords(new File(inFile_stopwords), "UTF-8", false, false, false) );
		            pipeList.add( new TokenSequence2FeatureSequence() );
		            
		            instances = new InstanceList (new  SerialPipes(pipeList));
		            
		            String pattern="(\\S{1,})[\\s+](\\S{1,})[\\s+]\"(.*?)\"[\\s+]\"(.*?)\""; 
		            System.out.println("pattern:"+pattern);
		            Reader fileReader = new InputStreamReader(new FileInputStream(new File(inFile)), "UTF-8");
		            instances.addThruPipe(new 	CsvIterator (fileReader,
		            							Pattern.compile(pattern),
		            							4, 3, 2, 1)); // title, data, label, name fields
		            
		            
		            
//            		Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(\\S*)[\\s,](.*)$"),
//        			Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
//            		Pattern.compile("^(\\S*)[oooo]*(\\S*)[oooo]*(\\S*)[oooo](.*)$"),
//            		Pattern.compile("(\\S+)*[\\t]*(\\S+)*[\\t]((\\w+\\s\\w){1,})"),
		            
		            //(\S+)*[\t]*(\S+)*[\t]((\w+\s\w){1,2})
		            
	            }
	             
	            // flag_loading_instances ==2
	            if(flag_loading_instances==2){
//	            	pipeList.add( new TokenSequence2FeatureSequence() );
//	            	pipeList.add( new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+")) );
//	            	pipeList.add(new TokenSequence2FeatureSequence());
//		            pipeList.add( new TokenSequenceRemoveStopwords(new File(inFile_stopwords), "UTF-8", false, false, false) );
//			        instances = new InstanceList (new  SerialPipes(pipeList));
  
	            	int[] gramSizes = {1};
	                pipeList.add( new SaveDataInSource() );
	                System.out.println("\n----after initialize 1 getDataAlphabet 1.1");
	                pipeList.add( new Input2CharSequence("STRING") );
	                System.out.println("\n----after initialize 1 getDataAlphabet 1.2");
	                pipeList.add (new CharSequence2TokenSequence());
	                System.out.println("\n----after initialize 1 getDataAlphabet 1.3");
	                pipeList.add( new TokenSequenceNGrams(gramSizes) );
	                System.out.println("\n----after initialize 1 getDataAlphabet 1.4");
	                pipeList.add( new TokenSequence2FeatureSequence() );
	                System.out.println("\n----after initialize 1 getDataAlphabet 1.5");
	                Pipe instancePipe = new SerialPipes(pipeList);
	                System.out.println("\n----after initialize 2 getDataAlphabet 1.6" );
	                instances = new InstanceList (instancePipe);
	                System.out.println("\n----after initialize 3 getDataAlphabet 1.7:"+instances.getDataAlphabet().size() );
	                
	            	// readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line
	            	map_lineNo_line =
				  				crawler.
					   	        readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile
					   	        .readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line(
					   	        													  inFile,
					   	        													  -1, -1,
					   									            				  "load  auth id & doc id"
					   									            				  , false //isPrintSOP
					   									            				  );
	            	// 
	            	for(int lineNo:map_lineNo_line.keySet()){
	            		System.out.println("------------------------------------------------");
	            		System.out.println("----after 2 initialize getDataAlphabet:"+instances.getDataAlphabet().size() );
	            		String currLine=map_lineNo_line.get(lineNo);
	            		currLine=currLine.replace("\t", "!#!#");
	            		String [] s=currLine.split(delimiter_of_inFile);
	            		System.out.println("s.len:"+s.length
	            							+" currLine:"+currLine
	            							+" delimiter:"+delimiter_of_inFile);
	            		//data, target, name, source, title
	            		Instance instance=new Instance(s[2], s[1], s[0], "", s[3]);
//	            		instances.add(instance); //NOT TRIED
	            		 
	            		System.out.println("getDataAlphabet:"+instances.getDataAlphabet().size());
	                    System.out.println("getTargetAlphabet:"+instances.getTargetAlphabet());
	                    
	            		instances.addThruPipe(instance); //WORKING
	            		
//	            		instances.addThruPipe(new StringArrayIterator(s));
	            	}
	            	
	            } // if(flag_loading_instances==2){
	            
	            String  inFile_authID_docIDCSV="";
	            String  inFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames="";
	            //
	            if(flag_loading_instances==3){

	        		//manually set 
	        		int index_of_data=3;
	                String 	pattern="(\\S{1,})[\\s+](\\S{1,})[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"";
	        				pattern="(\\S{1,})[\\s+](\\S{1,})[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"";//8 TOKENS
	        				pattern="(\\S{1,})[\\s+](\\S{1,})[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\""; //10	        				
	        				pattern="(\\S{1,})[\\s+](\\S{1,})[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\"[\\s+]\"(.*?)\""; //11
	        				
	        				
	        				
	        		inFile_authID_docIDCSV=baseFolder+"auth_id_doc_id.txt_NO_dirtyAuthNames.txt"; 
	        		inFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames=baseFolder+"auth_id_doc_id_queryTopicRelated.txt_NO_dirtyAuthNames.txt";
	        		String inFile_stopword="/Users/lenin/Dropbox/workspace-sts-3.4.0.RELEASE.macmini/grmm.malletBak/stopwords/en.txt";
	        		
	        		boolean isSOPprint=false;	            
	                // take only bodyText
	    	        instances_bodyText=ParallelTopicModel.wrapper_readFile_for_instance(	inFile, 
						    																inFile_stopword, 
						    																pattern,
						    																index_of_data, //bodyText 
						    																isSOPprint);
	        		num_of_documents=instances_bodyText.size();
	    	        System.out.println("------######### location----");
	    	        //take only numbers from here
	    	        instances_numbers=ParallelTopicModel.wrapper_readFile_for_instance( inFile,
	    															 inFile_stopword,
	    															 pattern,
	    															 4, //index_of_data,
	    															 isSOPprint);
	    	        System.out.println("------######### person ----");
	    	        //take only person+ location from here
	    	        instances_person_location=ParallelTopicModel.wrapper_readFile_for_instance(	
	    	        															inFile, 
	    																		inFile_stopword, 
	    																		pattern,
	    																		5, //index_of_data,
	    																		isSOPprint);
	    	        System.out.println("------######### organization----");
	    	        //take only organization
	    	        instances_organization=ParallelTopicModel.wrapper_readFile_for_instance( inFile,
						    																 inFile_stopword, 
						    																 pattern,
						    																 6, //index_of_data,
						    																 isSOPprint);
	    	        System.out.println("------######### fline----");
	    	        //take only first line (fline) of body from here
	    	        instances_fline=ParallelTopicModel.wrapper_readFile_for_instance(	inFile,
						    															inFile_stopword,
						    															pattern,
						    															7, //index_of_data,
						    															isSOPprint);
	    	        System.out.println("------######### title----");
	    	        //take only title from here
	    	        instances_title=ParallelTopicModel.wrapper_readFile_for_instance(		inFile,
						    																inFile_stopword,
						    																pattern,
						    																8, //index_of_data,
						    																isSOPprint);
	    	        //inFile->writing <docID\tlabel\tbodyText\tnumbers\tperson+location\torganization\tfirstline\ttitle\tkeywords>
	    	        System.out.println("------######### keywords----");
	    	        //take only keywords
	    	        instances_keywords=ParallelTopicModel.wrapper_readFile_for_instance(	inFile,
						    																inFile_stopword, 
						    																pattern,
						    																9, //index_of_data,
						    																isSOPprint);
	    	        System.out.println("------######### MISSING keywords----");
	    	        //take only MISSING keywords
	    	        instances_MISSINGkeywords=ParallelTopicModel.wrapper_readFile_for_instance(	inFile,
	    																		inFile_stopword, 
	    																		pattern,
	    																		10, //index_of_data,
	    																		isSOPprint);
	    	        System.out.println("------######### bool_TRADorPOLI----");
	    	        //take only bool_TRADorPOLI
	    	        instances_bool_TRADorPOLI=ParallelTopicModel.wrapper_readFile_for_instance(	inFile,
					    																		inFile_stopword, 
					    																		pattern,
					    																		11, //index_of_data,
					    																		isSOPprint);
	    	        System.out.println("------######### all features----");
	    	        instances_all_features=ParallelTopicModel.wrapper_readFile_for_instance(	inFile,
						    																	inFile_stopword, 
						    																	pattern,
						    																	12, //index_of_data,
						    																	isSOPprint);
	    	        
	    	        
	    	        System.out.println("------######### applied tags AND MISSING (related) keywords----");
	    	        instances_keywords_N__MISSINGkeywords=ParallelTopicModel.wrapper_readFile_for_instance(	inFile,
																											inFile_stopword, 
																											pattern,
																											13, //index_of_data, (13 token is primary key)
																											isSOPprint);
	            	
	            } //if(flag_loading_instances==3){
	            
				// Load auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames
				TreeMap<Integer, String> map_eachLine_inFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames=
						readFile_load_Each_Line_of_Generic_File_To_Map_String_String_remove_dup_write_to_outputFile.
						readFile_load_Each_Line_of_Generic_File_To_Map_Integer_Counter_String_Line( inFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames,
																									-1, 
																									-1,
																									 "load bodytext",//debug_label,
																									 false // isPrintSOP
																									 );
				int size_of_AuthID = map_eachLine_inFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames.size();
				
				
				if(run_including_LDA){
					
					// LDA
					SimpleLDA_ model = new SimpleLDA_( 	num_of_topics,
														50.0,   // alphaSum
														0.01,   // beta
														100 	// gamma{100}
													);
					
	//				System.out.println("dictionary-->"+instances_keywords_N__MISSINGkeywords.getAlphabet());
					// Adding Instance
					model.addInstances(
										instances_bodyText, //training
				    					instances_numbers,
				    					instances_person_location,
				    					instances_organization,
				    					instances_fline,
				    					instances_title,
				    				   	instances_keywords,
				    				    instances_MISSINGkeywords,
				    				    instances_bool_TRADorPOLI,
				    				    instances_all_features,
				    				    instances_keywords_N__MISSINGkeywords,
				    				   	writerDebug
									  );
					// print
					model.setTopicDisplay(25, // interval
										  100); //number of words				
					
					// lenin: refer http://www.jmlr.org/proceedings/papers/v13/xiao10a/xiao10a.pdf
					// http://www.ics.uci.edu/~asuncion/pubs/UAI_09.pdf
					// sample -- Initially 1000 , lenin changed to 100
					model.sample(	num_of_iterations, // iterations
									baseFolder,
									inFile_authID_docIDCSV,
									inFile_auth_id_doc_id_queryTopicRelated_NO_dirtyAuthNames,
									query_topicCSV, //OBSOLETE (used for Author)
									debugFile,
									false //is_SOPprint
								);
					int doc_index=0;
					// ------------------- INPUT TO GROUND TRUTH ground_truth.java
			        writer2_d_z_prob.append("d z p_z_d\n");
			        writer2_d_z_prob_TITLE.append("d z p_z_d\n");writer2_d_z_prob_AVG.append("d z p_z_d\n");
			        // WRITING TO output file --> d z p_z_d --- tokens from BODYTEXT
			        while(doc_index<num_of_documents){
		        		String conc_topic_prob="";String conc_topic_prob_TITLE="";
				        for (int topic = 0; topic < num_of_topics; topic++) {
	//			        		out = new Formatter(new StringBuilder(), Locale.US);
	//			        		System.out.println("doc_index:"+doc_index+" "+topic);
				        		// --------- BODYTEXT
				        		double currDocTopic_probability=model.getTopicProbabilities(doc_index)[topic];
			//		            out.format("print1: %d\t%.3f\t", topic, model.getTopicProbabilities(doc_index)[topic]);
				        		if(conc_topic_prob.length() ==0){
				        			conc_topic_prob=topic + ":"+ currDocTopic_probability;
				        		}
				        		else{
				        			conc_topic_prob=conc_topic_prob+" "+topic + ":"+ currDocTopic_probability;
				        		}
			        			// write "d z probability"
			        			writer2_d_z_prob.append(doc_index+" "+topic+" "+currDocTopic_probability +"\n");
			        			writer2_d_z_prob.flush();
			        			// --------- TITLE
			        			double currDocTopic_probability_TITLE=model.getTopicProbabilities_TITLE(doc_index)[topic];
			        			writer2_d_z_prob_TITLE.append(doc_index+" "+topic+" "+currDocTopic_probability_TITLE +"\n");
			        			writer2_d_z_prob_TITLE.flush();
			        			// --------- AVERAGE
			        			writer2_d_z_prob_AVG.append(doc_index+" "+topic+" "+(currDocTopic_probability+currDocTopic_probability_TITLE)/2. +"\n");
			        			writer2_d_z_prob_AVG.flush();
				        }
			            doc_index++;
	//		        	System.out.println("doc:"+doc_index+" "+conc_doc_prob);
			        	writer.append("doc:"+doc_index+" "+conc_topic_prob+"\n");
			        	writer.flush();
			        }
			        
					System.out.println("output file:"+debugFile);
					System.out.println("flag_loading_instances:"+flag_loading_instances);
					System.out.println("" );
				
				}
				random=new Randoms();
				//// 
				System.out.println("BELOW 3 lines for PAM MODEL");
				PAM4L model2=new PAM4L(6, 6) ;
				model2.estimate(instances_bodyText, 150, 10, 20, 20, baseFolder+"pam_MODEL.txt", random);
				model2.printDocumentTopics(new File(baseFolder+"pam_OUTPUT.txt"));
				  
				
				if(run_including_LDA==false)
					System.out.println("NOTE: flag 'run_including_LDA' is FALSE, so did not run LDA, for PAM running ,run class convert_PAM_output_to_d_by_z_probabilityFORMAT after this");
				
				System.out.println("\nTime Taken (FINAL ENDED):"
										+ NANOSECONDS.toSeconds(System.nanoTime() - t0)
										+ " seconds; "
										+ (NANOSECONDS.toSeconds(System.nanoTime() - t0)) / 60
										+ " minutes");
//				\p{L} matches a single code point in the category "letter".xx
//				\p{N} matches any kind of numeric character in any script.
		}
		catch(Exception e){
			System.out.println(" message; "+e.getMessage());
			e.printStackTrace(	);
		}
	}	// END public static void main(String[] args) throws IOException {
}