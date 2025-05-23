#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../dataio/sgf.h"
#include "../neuralnet/modelversion.h"
#include "../search/asyncbot.h"
#include "../search/searchnode.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../main.h"

using namespace std;

int MainCmds::evalsgf(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string modelFile;
  string humanModelFile;
  string sgfFile;
  int moveNumStart;
  int moveNumEnd;
  string printBranch;
  string extraMoves;
  string avoidMoves;
  string hintLoc;
  int64_t maxVisits;
  int numThreads;
  float overrideKomi;
  string overrideRules;
  bool printOwnership;
  bool printRootNNValues;
  bool printPolicy;
  bool printLogPolicy;
  bool printDirichletShape;
  bool printScoreNow;
  bool printRootEndingBonus;
  bool printLead;
  bool printAvgShorttermError;
  bool printSharpScore;
  bool printGraph;
  bool printJson;
  int printMaxDepth;
  bool rawNN;
  string dumpNpzInputTo;
  try {
    KataGoCommandLine cmd("Run a search on a position from an sgf file, for debugging.");
    cmd.addConfigFileArg("","gtp_example.cfg");
    cmd.addModelFileArg();
    cmd.addHumanModelFileArg();

    TCLAP::UnlabeledValueArg<string> sgfFileArg("","Sgf file to analyze",true,string(),"FILE");
    TCLAP::ValueArg<int> moveNumArg("m","move-num","Sgf move num to analyze",true,0,"MOVENUM");
    TCLAP::ValueArg<int> moveNumEndArg("","move-num-end","End sgf move num range to analyze, inclusive",false,-1,"MOVENUM");

    TCLAP::ValueArg<string> printBranchArg("","print-branch","Move branch in search tree to print",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> printArg("p","print","Alias for -print-branch",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> extraMovesArg("","extra-moves","Extra moves to force-play before doing search",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> extraArg("e","extra","Alias for -extra-moves",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> avoidMovesArg("","avoid-moves","Avoid moves in search",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> hintLocArg("","hint-loc","Hint loc",false,string(),"MOVE");
    TCLAP::ValueArg<long> visitsArg("v","visits","Set the number of visits",false,-1,"VISITS");
    TCLAP::ValueArg<int> threadsArg("t","threads","Set the number of threads",false,-1,"THREADS");
    TCLAP::ValueArg<float> overrideKomiArg("","override-komi","Artificially set komi",false,std::numeric_limits<float>::quiet_NaN(),"KOMI");
    TCLAP::ValueArg<string> overrideRulesArg("","override-rules","Artifically set rules",false,string(),"RULES");
    TCLAP::SwitchArg printOwnershipArg("","print-ownership","Print ownership");
    TCLAP::SwitchArg printRootNNValuesArg("","print-root-nn-values","Print root nn values");
    TCLAP::SwitchArg printPolicyArg("","print-policy","Print policy");
    TCLAP::SwitchArg printLogPolicyArg("","print-log-policy","Print log policy");
    TCLAP::SwitchArg printDirichletShapeArg("","print-dirichlet-shape","Print dirichlet shape");
    TCLAP::SwitchArg printScoreNowArg("","print-score-now","Print score now");
    TCLAP::SwitchArg printRootEndingBonusArg("","print-root-ending-bonus","Print root ending bonus now");
    TCLAP::SwitchArg printLeadArg("","print-lead","Compute and print lead");
    TCLAP::SwitchArg printAvgShorttermErrorArg("","print-avg-shortterm-error","Compute and print avgShorttermError");
    TCLAP::SwitchArg printSharpScoreArg("","print-sharp-score","Compute and print sharp weighted score");
    TCLAP::SwitchArg printGraphArg("","print-graph","Print graph structure of the search");
    TCLAP::SwitchArg printJsonArg("","print-json","Print analysis json of the search");
    TCLAP::ValueArg<int> printMaxDepthArg("","print-max-depth","How deep to print",false,1,"DEPTH");
    TCLAP::SwitchArg rawNNArg("","raw-nn","Perform single raw neural net eval");
    TCLAP::ValueArg<string> dumpNpzInputToArg("","dump-npz-input-to","Dump the nn input tensor to npz file",false,string(),"NPZFILE");
    cmd.add(sgfFileArg);
    cmd.add(moveNumArg);
    cmd.add(moveNumEndArg);

    cmd.setShortUsageArgLimit();

    cmd.addOverrideConfigArg();

    cmd.add(printBranchArg);
    cmd.add(printArg);
    cmd.add(extraMovesArg);
    cmd.add(extraArg);
    cmd.add(avoidMovesArg);
    cmd.add(hintLocArg);
    cmd.add(visitsArg);
    cmd.add(threadsArg);
    cmd.add(overrideKomiArg);
    cmd.add(overrideRulesArg);
    cmd.add(printOwnershipArg);
    cmd.add(printRootNNValuesArg);
    cmd.add(printPolicyArg);
    cmd.add(printLogPolicyArg);
    cmd.add(printDirichletShapeArg);
    cmd.add(printScoreNowArg);
    cmd.add(printRootEndingBonusArg);
    cmd.add(printLeadArg);
    cmd.add(printAvgShorttermErrorArg);
    cmd.add(printSharpScoreArg);
    cmd.add(printGraphArg);
    cmd.add(printJsonArg);
    cmd.add(printMaxDepthArg);
    cmd.add(rawNNArg);
    cmd.add(dumpNpzInputToArg);
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    humanModelFile = cmd.getHumanModelFile();
    sgfFile = sgfFileArg.getValue();
    moveNumStart = moveNumArg.getValue();
    moveNumEnd = moveNumEndArg.getValue();
    printBranch = printBranchArg.getValue();
    string print = printArg.getValue();
    extraMoves = extraMovesArg.getValue();
    string extra = extraArg.getValue();
    avoidMoves = avoidMovesArg.getValue();
    hintLoc = hintLocArg.getValue();
    maxVisits = (int64_t)visitsArg.getValue();
    numThreads = threadsArg.getValue();
    overrideKomi = overrideKomiArg.getValue();
    overrideRules = overrideRulesArg.getValue();
    printOwnership = printOwnershipArg.getValue();
    printRootNNValues = printRootNNValuesArg.getValue();
    printPolicy = printPolicyArg.getValue();
    printLogPolicy = printLogPolicyArg.getValue();
    printDirichletShape = printDirichletShapeArg.getValue();
    printScoreNow = printScoreNowArg.getValue();
    printRootEndingBonus = printRootEndingBonusArg.getValue();
    printLead = printLeadArg.getValue();
    printAvgShorttermError = printAvgShorttermErrorArg.getValue();
    printSharpScore = printSharpScoreArg.getValue();
    printGraph = printGraphArg.getValue();
    printJson = printJsonArg.getValue();
    printMaxDepth = printMaxDepthArg.getValue();
    rawNN = rawNNArg.getValue();
    dumpNpzInputTo = dumpNpzInputToArg.getValue();

    if(printBranch.length() > 0 && print.length() > 0) {
      cerr << "Error: -print-branch and -print both specified" << endl;
      return 1;
    }
    if(printBranch.length() <= 0)
      printBranch = print;

    if(extraMoves.length() > 0 && extra.length() > 0) {
      cerr << "Error: -extra-moves and -extra both specified" << endl;
      return 1;
    }
    if(extraMoves.length() <= 0)
      extraMoves = extra;

    if(moveNumEnd < moveNumStart)
      moveNumEnd = moveNumStart;

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  //Parse rules -------------------------------------------------------------------
  Rules defaultRules = Rules::getTrompTaylorish();
  Player perspective = Setup::parseReportAnalysisWinrates(cfg,P_BLACK);

  //Parse sgf file and board ------------------------------------------------------------------

  CompactSgf* sgf = CompactSgf::loadFile(sgfFile);

  Board board;
  Player nextPla;
  BoardHistory hist;

  auto setUpBoardUsingRules = [&board,&nextPla,&hist,overrideKomi,&sgf,&extraMoves](const Rules& initialRules, int moveNum) {
    sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);
    vector<Move>& moves = sgf->moves;

    if(!isnan(overrideKomi)) {
      if(overrideKomi > board.x_size * board.y_size + NNPos::KOMI_CLIP_RADIUS || overrideKomi < -board.x_size * board.y_size - NNPos::KOMI_CLIP_RADIUS)
        throw StringError("Invalid komi, too much greater than the area of the board");
      hist.setKomi(overrideKomi);
    }

    if(moveNum < 0)
      throw StringError("Move num " + Global::intToString(moveNum) + " requested but must be non-negative");
    if(moveNum > moves.size())
      throw StringError("Move num " + Global::intToString(moveNum) + " requested but sgf has only " + Global::int64ToString(moves.size()));

    sgf->playMovesTolerant(board,nextPla,hist,moveNum,false);

    vector<Loc> extraMoveLocs = Location::parseSequence(extraMoves,board);
    for(size_t i = 0; i<extraMoveLocs.size(); i++) {
      Loc loc = extraMoveLocs[i];
      if(!hist.isLegal(board,loc,nextPla)) {
        cerr << board << endl;
        cerr << "Extra illegal move for " << PlayerIO::colorToChar(nextPla) << ": " << Location::toString(loc,board) << endl;
        throw StringError("Illegal extra move");
      }
      hist.makeBoardMoveAssumeLegal(board,loc,nextPla,NULL);
      nextPla = getOpp(nextPla);
    }
  };

  Rules initialRules = sgf->getRulesOrWarn(
    defaultRules,
    [](const string& msg) { cout << msg << endl; }
  );
  if(overrideRules != "") {
    initialRules = Rules::parseRules(overrideRules);
  }

  // Set up once now for error catcihng
  setUpBoardUsingRules(initialRules,moveNumStart);

  //Parse move sequence arguments------------------------------------------

  PrintTreeOptions options;
  options = options.maxDepth(printMaxDepth);
  if(printBranch.length() > 0)
    options = options.onlyBranch(board,printBranch);
  options = options.printAvgShorttermError(printAvgShorttermError);

  //Load neural net and start bot------------------------------------------

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);
  logger.write("Engine starting...");

  bool hasHumanModel = humanModelFile != "";
  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP,hasHumanModel);
  if(maxVisits < -1 || maxVisits == 0)
    throw StringError("maxVisits: invalid value");
  else if(maxVisits == -1)
    logger.write("No max visits specified on cmdline, using defaults in " + cfg.getFileName());
  else {
    params.maxVisits = maxVisits;
    params.maxPlayouts = maxVisits; //Also set this so it doesn't cap us either
  }
  if(numThreads < -1 || numThreads == 0)
    throw StringError("numThreads: invalid value");
  else if(numThreads == -1)
    logger.write("No num threads specified on cmdline, using defaults in " + cfg.getFileName());
  else {
    params.numThreads = numThreads;
  }

  string searchRandSeed;
  if(cfg.contains("searchRandSeed"))
    searchRandSeed = cfg.getString("searchRandSeed");
  else
    searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());

  NNEvaluator* nnEval = NULL;
  NNEvaluator* humanEval = NULL;
  {
    Setup::initializeSession(cfg);
    int expectedConcurrentEvals = params.numThreads;
    int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
    bool defaultRequireExactNNLen = true;
    bool disableFP16 = false;
    string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
      board.x_size,board.y_size,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_GTP
    );
    if(humanModelFile != "") {
      humanEval = Setup::initializeNNEvaluator(
        humanModelFile,humanModelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
        board.x_size,board.y_size,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_GTP
      );
    }
  }
  logger.write("Loaded neural net");

  {
    bool rulesWereSupported;
    Rules supportedRules = nnEval->getSupportedRules(initialRules,rulesWereSupported);
    if(!rulesWereSupported) {
      cout << "Warning: Rules " << initialRules << " from sgf not supported by neural net, using " << supportedRules << " instead" << endl;
      initialRules = supportedRules;
    }
  }

  for(int moveNum = moveNumStart; moveNum <= moveNumEnd; moveNum++) {
    setUpBoardUsingRules(initialRules,moveNum);

    // {
    //   sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);
    //   vector<Move>& moves = sgf->moves;

    //   for(size_t i = 0; i<moves.size(); i++) {
    //     bool preventEncore = false;
    //     bool suc = hist.makeBoardMoveTolerant(board,moves[i].loc,moves[i].pla,preventEncore);
    //     assert(suc);
    //     nextPla = getOpp(moves[i].pla);

    //     MiscNNInputParams nnInputParams;
    //     nnInputParams.nnPolicyTemperature = 1.2f;
    //     NNResultBuf buf;
    //     bool skipCache = true;
    //     bool includeOwnerMap = false;
    //     nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

    //     NNOutput* nnOutput = buf.result.get();
    //     vector<double> probs;
    //     for(int y = 0; y<board.y_size; y++) {
    //       for(int x = 0; x<board.x_size; x++) {
    //         int pos = NNPos::xyToPos(x,y,nnOutput->nnXLen);
    //         float prob = nnOutput->policyProbs[pos];
    //         probs.push_back(prob);
    //       }
    //     }
    //     std::sort(probs.begin(),probs.end());
    //     cout << probs[probs.size()-1] << " " << probs[probs.size()-2] << " " << probs[probs.size()-3] << endl;
    //   }
    //   continue;
    // }

    // {
    //   sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);
    //   vector<Move>& moves = sgf->moves;

    //   for(size_t i = 0; i<moves.size(); i++) {
    //     bool preventEncore = false;
    //     bool suc = hist.makeBoardMoveTolerant(board,moves[i].loc,moves[i].pla,preventEncore);
    //     assert(suc);
    //     nextPla = getOpp(moves[i].pla);

    //     MiscNNInputParams nnInputParams;
    //     nnInputParams.drawEquivalentWinsForWhite = params.drawEquivalentWinsForWhite;
    //     NNResultBuf buf;
    //     bool skipCache = true;
    //     bool includeOwnerMap = false;
    //     nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

    //     NNOutput* nnOutput = buf.result.get();
    //     cout << nnOutput->whiteWinProb << " " << nnOutput->shorttermWinlossError << " "
    //          << nnOutput->whiteScoreMean << " " << nnOutput->shorttermScoreError  << endl;
    //   }
    //   continue;
    // }

    //Check for unused config keys
    cfg.warnUnusedKeys(cerr,&logger);
    Setup::maybeWarnHumanSLParams(params,nnEval,NULL,cerr,&logger);

    if(rawNN) {
      NNResultBuf buf;
      bool skipCache = true;
      bool includeOwnerMap = true;
      MiscNNInputParams nnInputParams;
      nnInputParams.drawEquivalentWinsForWhite = params.drawEquivalentWinsForWhite;
      nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

      cout << "Rules: " << hist.rules << endl;
      cout << "Encore phase " << hist.encorePhase << endl;
      Board::printBoard(cout, board, Board::NULL_LOC, &(hist.moveHistory));
      buf.result->debugPrint(cout,board);

      if(humanEval != NULL) {
        humanEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);
        buf.result->debugPrint(cout,board);
      }
      continue;
    }

    AsyncBot* bot = new AsyncBot(params, nnEval, humanEval, &logger, searchRandSeed);

    bot->setPosition(nextPla,board,hist);
    if(hintLoc != "") {
      bot->setRootHintLoc(Location::ofString(hintLoc,board));
    }

    if(avoidMoves != "") {
      vector<Loc> avoidMoveLocs = Location::parseSequence(avoidMoves,board);
      vector<int> avoidMoveUntilByLoc(Board::MAX_ARR_SIZE,0);
      for(Loc loc: avoidMoveLocs)
        avoidMoveUntilByLoc[loc] = 1;
      bot->setAvoidMoveUntilByLoc(avoidMoveUntilByLoc,avoidMoveUntilByLoc);
    }

    //Print initial state----------------------------------------------------------------
    const Search* search = bot->getSearchStopAndWait();
    ostringstream sout;
    sout << "Rules: " << hist.rules << endl;
    sout << "Encore phase " << hist.encorePhase << endl;
    Board::printBoard(sout, board, Board::NULL_LOC, &(hist.moveHistory));

    if(options.branch_.size() > 0) {
      Board copy = board;
      BoardHistory copyHist = hist;
      Player pla = nextPla;
      for(int i = 0; i<options.branch_.size(); i++) {
        Loc loc = options.branch_[i];
        if(!copyHist.isLegal(copy,loc,pla)) {
          cerr << board << endl;
          cerr << "Branch Illegal move for " << PlayerIO::colorToChar(pla) << ": " << Location::toString(loc,board) << endl;
          return 1;
        }
        copyHist.makeBoardMoveAssumeLegal(copy,loc,pla,NULL);
        pla = getOpp(pla);
      }
      Board::printBoard(sout, copy, Board::NULL_LOC, &(copyHist.moveHistory));
    }

    sout << "\n";
    logger.write(sout.str());
    sout.clear();

    //Search!----------------------------------------------------------------

    ClockTimer timer;
    nnEval->clearStats();
    if(humanEval != NULL)
      humanEval->clearStats();
    Loc loc = bot->genMoveSynchronous(bot->getSearch()->rootPla,TimeControls());
    (void)loc;

    //Postprocess------------------------------------------------------------

    if(printOwnership) {
      sout << "Ownership map (ROOT position):\n";
      search->printRootOwnershipMap(sout,perspective);
    }

    if(printRootNNValues) {
      const NNOutput* nnOutput = search->rootNode->getNNOutput();
      if(nnOutput != NULL) {
        cout << "White win: " << nnOutput->whiteWinProb << endl;
        cout << "White loss: " << nnOutput->whiteLossProb << endl;
        cout << "White noresult: " << nnOutput->whiteNoResultProb << endl;
        cout << "White score mean " << nnOutput->whiteScoreMean << endl;
        cout << "White score stdev " << sqrt(max(0.0,(double)nnOutput->whiteScoreMeanSq - nnOutput->whiteScoreMean*nnOutput->whiteScoreMean)) << endl;
        cout << "Var time left " << nnOutput->varTimeLeft << endl;
        cout << "Shortterm winloss error " << nnOutput->shorttermWinlossError << endl;
        cout << "Shortterm score error " << nnOutput->shorttermScoreError << endl;
      }
    }

    // {
    //   ReportedSearchValues values;
    //   bool suc = search->getRootValues(values);
    //   if(!suc)
    //     cout << "Unsuccessful getting root values" << endl;
    //   else
    //     cout << values << endl;
    // }
    // {
    //   ReportedSearchValues values;
    //   bool suc = search->getPrunedRootValues(values);
    //   if(!suc)
    //     cout << "Unsuccessful getting pruned root values" << endl;
    //   else
    //     cout << values << endl;
    // }


    if(printSharpScore) {
      double ret = 0.0;
      bool suc = search->getSharpScore(NULL,ret);
      assert(suc);
      (void)suc;
      cout << "White sharp score " << ret << endl;
    }

    if(printPolicy) {
      auto doPrintPolicy = [&](const float* policyProbs, int nnXLen, int nnYLen) {
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            int pos = NNPos::xyToPos(x,y,nnXLen);
            double prob = policyProbs[pos];
            if(prob < 0)
              cout << "  -  " << " ";
            else
              cout << Global::strprintf("%5.2f",prob*100) << " ";
          }
          cout << endl;
        }
        double prob = policyProbs[NNPos::locToPos(Board::PASS_LOC,board.x_size,nnXLen,nnYLen)];
        cout << "Pass " << Global::strprintf("%5.2f",prob*100) << endl;
      };

      const NNOutput* nnOutput = search->rootNode->getNNOutput();
      if(nnOutput != NULL) {
        const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
        cout << "Root policy: " << endl;
        doPrintPolicy(policyProbs, nnOutput->nnXLen, nnOutput->nnYLen);
      }
      const NNOutput* humanOutput = search->rootNode->getHumanOutput();
      if(humanOutput != NULL) {
        const float* policyProbs = humanOutput->getPolicyProbsMaybeNoised();
        cout << "Root human policy: " << endl;
        doPrintPolicy(policyProbs, humanOutput->nnXLen, humanOutput->nnYLen);
      }
    }
    if(printLogPolicy) {
      auto doPrintLogPolicy = [&](const float* policyProbs, int nnXLen, int nnYLen) {
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            int pos = NNPos::xyToPos(x,y,nnXLen);
            double prob = policyProbs[pos];
            if(prob < 0)
              cout << "  _  " << " ";
            else
              cout << Global::strprintf("%+5.2f",log(prob)) << " ";
          }
          cout << endl;
        }
        double prob = policyProbs[NNPos::locToPos(Board::PASS_LOC,board.x_size,nnXLen,nnYLen)];
        cout << "Pass " << Global::strprintf("%+5.2f",log(prob)) << endl;
      };

      const NNOutput* nnOutput = search->rootNode->getNNOutput();
      if(nnOutput != NULL) {
        const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
        cout << "Root policy: " << endl;
        doPrintLogPolicy(policyProbs, nnOutput->nnXLen, nnOutput->nnYLen);
      }
      const NNOutput* humanOutput = search->rootNode->getHumanOutput();
      if(humanOutput != NULL) {
        const float* policyProbs = humanOutput->getPolicyProbsMaybeNoised();
        cout << "Root human policy: " << endl;
        doPrintLogPolicy(policyProbs, humanOutput->nnXLen, humanOutput->nnYLen);
      }
    }

    if(printDirichletShape) {
      const NNOutput* nnOutput = search->rootNode->getNNOutput();
      if(nnOutput != NULL) {
        const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
        double alphaDistr[NNPos::MAX_NN_POLICY_SIZE];
        int policySize = nnOutput->nnXLen * nnOutput->nnYLen;
        Search::computeDirichletAlphaDistribution(policySize, policyProbs, alphaDistr);
        cout << "Dirichlet alphas with 10.83 total concentration: " << endl;
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            int pos = NNPos::xyToPos(x,y,nnOutput->nnXLen);
            double alpha = alphaDistr[pos];
            if(alpha < 0)
              cout << "  -  " << " ";
            else
              cout << Global::strprintf("%5.4f",alpha * 10.83) << " ";
          }
          cout << endl;
        }
        double alpha = alphaDistr[NNPos::locToPos(Board::PASS_LOC,board.x_size,nnOutput->nnXLen,nnOutput->nnYLen)];
        cout << "Pass " << Global::strprintf("%5.2f",alpha * 10.83) << endl;
      }
    }

    if(printScoreNow) {
      sout << "Score now (ROOT position):\n";
      Board copy(board);
      BoardHistory copyHist(hist);
      Color area[Board::MAX_ARR_SIZE];
      copyHist.endAndScoreGameNow(copy,area);

      for(int y = 0; y<copy.y_size; y++) {
        for(int x = 0; x<copy.x_size; x++) {
          Loc l = Location::getLoc(x,y,copy.x_size);
          sout << PlayerIO::colorToChar(area[l]);
        }
        sout << endl;
      }
      sout << endl;

      sout << "Komi: " << copyHist.rules.komi << endl;
      sout << "WBonus: " << copyHist.whiteBonusScore << endl;
      sout << "Final: "; WriteSgf::printGameResult(sout, copyHist); sout << endl;
    }

    if(printRootEndingBonus) {
      sout << "Ending bonus (ROOT position)\n";
      search->printRootEndingScoreValueBonus(sout);
    }

    sout << "Time taken: " << timer.getSeconds() << "\n";
    sout << "Root visits: " << search->getRootVisits() << "\n";
    sout << "NN rows: " << nnEval->numRowsProcessed() << endl;
    sout << "NN batches: " << nnEval->numBatchesProcessed() << endl;
    sout << "NN avg batch size: " << nnEval->averageProcessedBatchSize() << endl;
    std::vector<SearchNode*> nodes = bot->getSearchStopAndWait()->enumerateTreePostOrder();
    sout << "True number of tree nodes: " << nodes.size() << endl;
    sout << "PV: ";
    search->printPV(sout, search->rootNode, 25);
    sout << "\n";
    sout << "Tree:\n";
    search->printTree(sout, search->rootNode, options, perspective);
    logger.write(sout.str());

    if(printLead) {
      BoardHistory hist2(hist);
      double lead = PlayUtils::computeLead(
        bot->getSearchStopAndWait(), NULL, board, hist2, nextPla,
        20, OtherGameProperties()
      );
      cout << "LEAD: " << lead << endl;
    }

    if(printGraph) {
      std::reverse(nodes.begin(),nodes.end());
      std::map<SearchNode*,size_t> idxOfNode;
      for(size_t nodeIdx = 0; nodeIdx<nodes.size(); nodeIdx++)
        idxOfNode[nodes[nodeIdx]] = nodeIdx;

      for(int nodeIdx = 0; nodeIdx<nodes.size(); nodeIdx++) {
        SearchNode& node = *(nodes[nodeIdx]);
        SearchNodeChildrenReference children = node.getChildren();
        int childrenCapacity = children.getCapacity();
        for(int i = 0; i<childrenCapacity; i++) {
          SearchNode* child = children[i].getIfAllocated();
          if(child == NULL)
            break;
          cout << nodeIdx << " -> " << idxOfNode[child] << "\n";
        }
      }
      cout << endl;
    }

    if(printJson) {
      int analysisPVLen = 7;
      bool preventEncore = false;
      bool includePolicy = printPolicy;
      bool includeOwnership = printOwnership;
      bool includeOwnershipStdev = false;
      bool includeMovesOwnership = false;
      bool includeMovesOwnershipStdev = false;
      bool includePVVisits = true;
      nlohmann::json ret;
      bool suc = search->getAnalysisJson(
        perspective,
        analysisPVLen,
        preventEncore,
        includePolicy,
        includeOwnership,
        includeOwnershipStdev,
        includeMovesOwnership,
        includeMovesOwnershipStdev,
        includePVVisits,
        ret
      );
      if(suc) {
        cout << ret << endl;
      }
    }

    if(dumpNpzInputTo != "") {
      bool inputsUseNHWC = false;
      int nnXLen = nnEval->getNNXLen();
      int nnYLen = nnEval->getNNYLen();
      int modelVersion = nnEval->getModelVersion();
      int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
      int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);

      NumpyBuffer<float> binaryInputNCHW(std::vector<int64_t>({1,numSpatialFeatures,nnXLen,nnYLen}));
      NumpyBuffer<float> globalInputNC(std::vector<int64_t>({1,numGlobalFeatures}));

      MiscNNInputParams nnInputParams;
      nnInputParams.symmetry = 0;
      nnInputParams.policyOptimism = params.rootPolicyOptimism;
      NNInputs::fillRowV7(board, hist, nextPla, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, binaryInputNCHW.data, globalInputNC.data);

      ZipFile zipFile(dumpNpzInputTo);
      uint64_t numBytes;

      numBytes = binaryInputNCHW.prepareHeaderWithNumRows(1);
      zipFile.writeBuffer("binaryInputNCHW", binaryInputNCHW.dataIncludingHeader, numBytes);
      numBytes = globalInputNC.prepareHeaderWithNumRows(1);
      zipFile.writeBuffer("globalInputNC", globalInputNC.dataIncludingHeader, numBytes);
      zipFile.close();
      cout << "Wrote to " << dumpNpzInputTo << endl;

      NNResultBuf buf;
      bool skipCache = true;
      bool includeOwnerMap = true;
      nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);
      buf.result->debugPrint(cout,board);
    }

    delete bot;
  }

  delete nnEval;
  if(humanEval != NULL)
    delete humanEval;
  NeuralNet::globalCleanup();
  delete sgf;
  ScoreValue::freeTables();

  return 0;
}
