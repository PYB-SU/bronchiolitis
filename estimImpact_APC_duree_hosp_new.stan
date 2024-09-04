// in this model IE() depends on time since immunization, computed from mid-September

data {
  int nHopital;
  int nSeason; // not Including Season 2023
  int nAge;
  int nWeek;
  int offsetIE; //do we assume start from September 
  // this is for this year
  // the first HOSPITAL is TROUSSEAU
  array[nHopital,nWeek,nAge] int<lower=0> NBRO; // nombre de Bronchiolites par quinzaine et age
  // this is for past Seasons
  array[nSeason,nHopital,nWeek,nAge] int<lower=0> NREF; // nombre des années passées

  //same for hospitalisation
  array[nHopital,nWeek,nAge] int<lower=0> NHOSPBRO; // nombre de hospitalisations par quinzaine et age
  array[nSeason,nHopital,nWeek,nAge] int<lower=0> NHOSPREF; // nombre de hospitalisations passées
  
  // this is only TROUSSEAU
  // to compute coverage -> NOBRO_BEY, NO_BRO_NOBEY
  array[nWeek,nAge] int NOBRO_BEY;
  array[nWeek,nAge] int NOBRO_NOBEY;

  // for nirsevimab in hosp or not
  array[nWeek,nAge] int BRO_BEY;
  array[nWeek,nAge] int BRO_NOBEY;

  // for hosp in bey or not
  array[nWeek,nAge] int BRO_BEY_HOSP;
  array[nWeek,nAge] int BRO_NOBEY_HOSP;

  //VRS BEY among BRO & HOSP
  array[nWeek,nAge] int VRS_HOSP_BEY;
  array[nWeek,nAge] int VRS_HOSP_NOBEY;
  array[nWeek,nAge] int NPCR_HOSP_BEY;
  array[nWeek,nAge] int NPCR_HOSP_NOBEY;
  
  // commutateur modele
  int hospDepTime; // 0 si hospitalisation depend du temps; 1 sinon

  // for COUNTERFACTUAL
  int nScenario;
  vector[nScenario] pBeyScenario;
}

parameters {
  vector[nAge+nWeek-1] logit_pBeyfortus; //coverage
  vector<lower=-0.1, upper=1>[nWeek+offsetIE] IE; //efficacy in those vaccinated at birth

  // this here to model past Seasons - bronchiolitis
  matrix[nSeason,nWeek-1] muSeasonTime; // ageRatio
  matrix[nHopital,nAge-1] muHopitalAge; // ageRatio - this will be shared
  matrix[nSeason,nHopital] muSeasonHopital; // seasonHopital / 

  // 
  // this here to model past Seasons - bronchiolitis
  matrix[hospDepTime==1?nSeason:0,hospDepTime==1?(nWeek-1):0] deltaSeasonTime; // ageRatio
  matrix[nHopital,nAge-1] deltaHopitalAge; // ageRatio - this will be shared
  matrix[nSeason,nHopital] deltaSeasonHopital; // seasonHopital / 

  real logit_nu; //multiplyer for hospitalisation in the nirsevimab+
  
  // this to analyse current season
  vector [nHopital] mu2023Hopital; // this is intercept for Niversimab year
  vector[nWeek-1] muTime;
  
  // this to analyse current season
  vector [nHopital] delta2023Hopital; // this is intercept for Niversimab year
  vector[nWeek-1] deltaTime;

  // proportion VRS is expit(log_OR_VRS_Age+log_OR_Week)
  vector[nAge-1] log_beta_VRS_age; // lambda_V = lambda_All * prop_VRS
  vector[nWeek-1] log_beta_VRS_week; // lambda_V = lambda_All * prop_VRS
  vector[nHopital] z_log_beta_VRS_hosp; //to modify pVRS
  real mean_log_beta_VRS_hosp;
  
  real<lower=0> sdpBeyfortus; // sd of pBeyfortus change
  real<lower=0> sdIE;
 
  real<lower=0> sdMuSeasonHopital;
  real<lower=0> sdMuHopitalAge;
  real<lower=0> sdMuSeasonTime;

  real<lower=0> sdDeltaSeasonHopital;
  real<lower=0> sdDeltaHopitalAge;
  real<lower=0> sdDeltaSeasonTime;

  real<lower=0> sdBetaAge;
  real<lower=0> sdBetaTime;
  real<lower=0> sdlogBetaVRSHosp;
  real<lower=0> sdlogitpHopitalAge;
}

model {
  // distribution age
  vector[nAge+nWeek-1] pBeyfortus;
  pBeyfortus=inv_logit(logit_pBeyfortus); ///warning this is reversed, from pBeyfortus[nAge+nWeek-1 to 1]

  // proportion Beyfortus 
  matrix[nWeek,nAge] pBEY_BRO;
  matrix[nWeek,nAge] pHOSP_BEY_BRO;
  matrix[nWeek,nAge] pHOSP_NOBEY_BRO;

  // mean number of cases per week and age in each hopital * season
  matrix[nWeek,nAge]lambda_BRO;
  matrix[nWeek,nAge]p_HOSP;
  // mean
  matrix[nWeek,nAge] pVRS_BEY;
  matrix[nWeek,nAge] pVRS_NO_BEY;
  
  vector[nHopital] log_beta_VRS_hosp;
  
  log_beta_VRS_hosp = mean_log_beta_VRS_hosp + sdlogBetaVRSHosp * z_log_beta_VRS_hosp;
  
  real nu;
  nu=inv_logit(logit_nu);
  
  real pVRS;
  //Trousseau
  //copy log_beta_VRSAge and log_beta_VRS_time to vectors
  
  for (i in 1:nWeek) {
      for (j in 1:nAge) {
       
         //VRS chez les beyfortus
         if (i==1) { //reference is i==1 for week
           if (j==1){ //reference is j==1 for age
             pVRS = inv_logit(log_beta_VRS_hosp[1]); //reference
           } else {
             pVRS = inv_logit(log_beta_VRS_hosp[1]+log_beta_VRS_age[j-1]); //age
           }
         } else {
           if (j==1){
             pVRS = inv_logit(log_beta_VRS_hosp[1]+log_beta_VRS_week[i-1]); //week
           } else {
             pVRS = inv_logit(log_beta_VRS_hosp[1]+log_beta_VRS_age[j-1]+log_beta_VRS_week[i-1]); //week and age
           }
         }
         // duree 
        int d;
        if (j <= i + offsetIE) {
          d=j;//duree egale age si ne apres septembre
        } else {
          d=i+offsetIE;
        }
        
         pVRS_BEY[i,j] = pVRS*(1.0-IE[d]) / ( 1.0-pVRS*IE[d]);  //age and time influence simplifies
         VRS_HOSP_BEY[i,j] ~ binomial(NPCR_HOSP_BEY[i,j], pVRS_BEY[i,j]);
      
        //VRS chez les non beyfortus
         pVRS_NO_BEY[i,j] = pVRS;  //age and time influence simplifies
         VRS_HOSP_NOBEY[i,j] ~ binomial(NPCR_HOSP_NOBEY[i,j], pVRS_NO_BEY[i,j]);

       //coverage chez les non bronchiolite
        NOBRO_BEY[i,j] ~ binomial(NOBRO_BEY[i,j] + NOBRO_NOBEY[i,j], pBeyfortus[i-j+nAge]);

//       coverage chez les bronchiolites
        pBEY_BRO[i,j] = (pBeyfortus[i-j+nAge]*(1.0-pVRS*IE[d]))/(1.0-pBeyfortus[i-j+nAge]*pVRS*IE[d]);  //age and time influence simplifies
        BRO_BEY[i,j] ~ binomial(BRO_BEY[i,j]+ BRO_NOBEY[i,j], pBEY_BRO[i,j]);
        
//      hospitalisation chez les bronchiolites immunisées
//        pHOSP_BEY_BRO[i,j] = nu*rho;
//        BRO_BEY_HOSP[i,j] ~ binomial(BRO_BEY[i,j], pHOSP_BEY_BRO[i,j]);
        // hospitalisation chez les bro non immunisés
//        pHOSP_NOBEY_BRO[i,j] = rho;
//        BRO_NOBEY_HOSP[i,j] ~ binomial(BRO_NOBEY[i,j], pHOSP_NOBEY_BRO[i,j]);
      }    //
  }

  for (i in 3:(nAge+nWeek-1)) { //AR for pBeyfortus
    logit_pBeyfortus[i]~normal(2*logit_pBeyfortus[i-1]-logit_pBeyfortus[i-2],sdpBeyfortus);
  }
  logit_pBeyfortus[2]~normal(logit_pBeyfortus[1],sdpBeyfortus); //smallvalueat the beginning
  logit_pBeyfortus[1]~normal(-5,sdpBeyfortus); //smallvalueat the beginning
  
  for (i in 3:(nWeek+offsetIE)) { //AR for IE
    IE[i]~normal(2.0*IE[i-1]-IE[i-2],sdIE);
  }
  IE[2]~normal(IE[1],sdIE);
  IE[1]~normal(0,sdIE);
  

  
  // Hospitals for 2023
  for (k in 1:nHopital) {
    // total number is Poisson
     for (i in 1:nWeek) {
       for (j in 1:nAge) {
          //incidence of Bronchiolitis in age class
           // duree 
        int d;
        if (j <= i + offsetIE) {
          d=j; //duree egale age si ne apres septembre
        } else {
          d=i+offsetIE;
        }
         if (i==1) {
           if (j==1) {
             pVRS=inv_logit(log_beta_VRS_hosp[k]);
             lambda_BRO[i,j] = exp(mu2023Hopital[k])*(1.0-pVRS*pBeyfortus[i-j+nAge] * IE[d]);
             p_HOSP[i,j] = (inv_logit(delta2023Hopital[k]+logit_nu)*pBeyfortus[i-j+nAge]*(1.0-pVRS * IE[d]) +
                                                         inv_logit(delta2023Hopital[k])*(1.0-pBeyfortus[i-j+nAge]))/
                                                         (1.0-pVRS*pBeyfortus[i-j+nAge] * IE[d]);

           }else {
             pVRS=inv_logit(log_beta_VRS_hosp[k]+log_beta_VRS_age[j-1]);
             lambda_BRO[i,j] = exp(mu2023Hopital[k]+muHopitalAge[k,j-1]) *(1.0-pVRS*pBeyfortus[i-j+nAge] * IE[d]);
             p_HOSP[i,j] = (inv_logit(delta2023Hopital[k]+deltaHopitalAge[k,j-1]+logit_nu)*pBeyfortus[i-j+nAge]*(1.0-pVRS * IE[d]) +
                                                         inv_logit(delta2023Hopital[k]+deltaHopitalAge[k,j-1])*(1.0-pBeyfortus[i-j+nAge]))/
                                                         (1.0-pVRS*pBeyfortus[i-j+nAge] * IE[d]);}
         } else {
           if (j==1) {
             pVRS=inv_logit(log_beta_VRS_hosp[k]+log_beta_VRS_week[i-1]);
             lambda_BRO[i,j] = exp(mu2023Hopital[k] + muTime[i-1])*(1.0-pVRS*pBeyfortus[i-j+nAge] * IE[d]);
             p_HOSP[i,j] = (inv_logit(delta2023Hopital[k]+deltaTime[i-1]+logit_nu)*pBeyfortus[i-j+nAge]*(1.0-pVRS * IE[d]) +
                                                         inv_logit(delta2023Hopital[k]+deltaTime[i-1])*(1.0-pBeyfortus[i-j+nAge]))/
                                                         (1.0-pVRS*pBeyfortus[i-j+nAge] * IE[d]);
           } else {
             pVRS=inv_logit(log_beta_VRS_hosp[k]+log_beta_VRS_age[j-1]+log_beta_VRS_week[i-1]);
             lambda_BRO[i,j] = exp(mu2023Hopital[k] + muHopitalAge[k,j-1] + muTime[i-1])*(1.0-pVRS*pBeyfortus[i-j+nAge] * IE[d]);
             p_HOSP[i,j] = (inv_logit(delta2023Hopital[k]+deltaHopitalAge[k,j-1]+deltaTime[i-1]+logit_nu)*pBeyfortus[i-j+nAge]*(1.0-pVRS * IE[d]) +
                                                         inv_logit(delta2023Hopital[k]+deltaHopitalAge[k,j-1]+deltaTime[i-1])*(1.0-pBeyfortus[i-j+nAge]))/
                                                         (1.0-pVRS*pBeyfortus[i-j+nAge] * IE[d]);
           }
         }

         NBRO[k,i,j] ~ poisson(lambda_BRO[i,j]);
         NHOSPBRO[k,i,j] ~ binomial(NBRO[k,i,j], p_HOSP[i,j]);
      }
     }
  }

  for (i in 2:(nWeek-1)) {
    muTime[i]~normal(muTime[i-1],sdMuSeasonTime);
    deltaTime[i]~normal(deltaTime[i-1],sdDeltaSeasonTime);
  }
  muTime[1] ~ normal(0,sdMuSeasonTime);
  deltaTime[1] ~ normal(0,sdDeltaSeasonTime);


  // past Seasons
  for (l in 1:nSeason) {
    for (k in 1:nHopital) {
       for (i in 1:nWeek) {
         for (j in 1:nAge) {
           // incidence of Bronchiolitis/ we put 0 in first age and first Time
           if (i==1) {
             if (j==1) {
               lambda_BRO[i,j] = exp(muSeasonHopital[l,k]);
               p_HOSP[i,j] = inv_logit(deltaSeasonHopital[l,k]);
             } else {
               lambda_BRO[i,j] = exp(muSeasonHopital[l,k] + muHopitalAge[k,j-1]);
               p_HOSP[i,j] = inv_logit(deltaSeasonHopital[l,k]+deltaHopitalAge[k,j-1]);
             }
           } else {
             if(j==1) {
               lambda_BRO[i,j] = exp(muSeasonHopital[l,k] + muSeasonTime[l,i-1]);
               p_HOSP[i,j] = inv_logit(deltaSeasonHopital[l,k]+deltaSeasonTime[l,i-1]);
             } else {
               lambda_BRO[i,j] = exp(muSeasonHopital[l,k] + muSeasonTime[l,i-1] + muHopitalAge[k,j-1]);
               p_HOSP[i,j] = inv_logit(deltaSeasonHopital[l,k]+deltaSeasonTime[l,i-1] +deltaHopitalAge[k,j-1]);
             } 
           }
           NREF[l,k,i,j] ~poisson(lambda_BRO[i,j]);
           NHOSPREF[l,k,i,j] ~binomial(NREF[l,k,i,j], p_HOSP[i,j]);
         }
       }
    }
  }
  
  // profil Age in Hopital
  for (k in 1:nHopital) {
    for (i in 2:(nAge-1)) {
      muHopitalAge[k,i]~normal(muHopitalAge[k,i-1], sdMuHopitalAge);
      deltaHopitalAge[k,i]~normal(deltaHopitalAge[k,i-1], sdDeltaHopitalAge);
    }
    muHopitalAge[k,1] ~normal(0,sdMuHopitalAge);
    deltaHopitalAge[k,1] ~normal(0,sdDeltaHopitalAge);
  }
  sdMuHopitalAge~exponential(0.01);
  sdDeltaHopitalAge~exponential(0.01);
  
  // profil Time by Season
  for(l in 1:nSeason) {
    for (i in 2:(nWeek-1)) {
      muSeasonTime[l,i]~normal(muSeasonTime[l,i-1], sdMuSeasonTime);
      deltaSeasonTime[l,i]~normal(deltaSeasonTime[l,i-1], sdDeltaSeasonTime);
    }
    muSeasonTime[l,1] ~normal(0,sdMuSeasonTime);
    deltaSeasonTime[l,1] ~normal(0,sdDeltaSeasonTime);
  }
  sdMuSeasonTime~exponential(0.01);
  sdDeltaSeasonTime~exponential(0.01);
 
 //intercept season*hopital
 for (l in 1:nSeason) {
   for (k in 1:nHopital) {
     muSeasonHopital[l,k] ~normal(0,sdMuSeasonHopital);
     deltaSeasonHopital[l,k] ~normal(0,sdDeltaSeasonHopital);
   }
 }
 sdMuSeasonHopital~exponential(0.01);
 sdDeltaSeasonHopital~exponential(0.01);
 
  // intercept for year 2023
  for (i in 1:nHopital) {
   mu2023Hopital[i] ~normal(0,sdMuSeasonHopital);
   delta2023Hopital[i] ~normal(0,sdDeltaSeasonHopital);
  }
 
  //pVRS for age
  for (i in 3:(nAge-1)) {
     log_beta_VRS_age[i] ~normal(2*log_beta_VRS_age[i-1]+log_beta_VRS_age[i-2], sdBetaAge);
  } 
  log_beta_VRS_age[2] ~normal(log_beta_VRS_age[1], sdBetaAge);
  log_beta_VRS_age[1] ~normal(0, sdBetaAge);

  sdBetaAge~exponential(0.01);

  //pVRS for time
  for (i in 2:(nWeek-1)) {
     log_beta_VRS_week[i] ~normal(log_beta_VRS_week[i-1], sdBetaTime);
  } 
  log_beta_VRS_week[1] ~normal(0, sdBetaTime);
  
  sdBetaTime~exponential(0.01);
  
  // this is for change in pVRS among hospitals in 2023
  mean_log_beta_VRS_hosp~normal(0,1);
  sdlogBetaVRSHosp~exponential(1);
  // non centered parametrization
  z_log_beta_VRS_hosp~std_normal();
  
}

generated quantities {
  // incidences
  array[nScenario,nHopital,nWeek,nAge] real NCON;
  array[nHopital,nWeek,nAge] real NOBS;
  array[nHopital,nWeek,nAge] real pVRS;
  array[nScenario,nHopital,nWeek,nAge] real NHOSPCON;
  array[nHopital,nWeek,nAge] real NHOSPOBS;
  array[nHopital,nWeek,nAge] real pHOSP;
  // past
  array[nSeason,nHopital,nWeek,nAge] real NPASTBRO;
  array[nSeason,nHopital,nWeek,nAge] real NPASTHOSP;
  // impact
  matrix[nScenario, nHopital] impactBRO;
  matrix[nScenario, nHopital] impactHOSP;
  
  real denum;
  vector[nScenario] num;
  
  real denumHOSP;
  vector[nScenario] numHOSP;

  real nu;
  nu=inv_logit(logit_nu);

  vector[nHopital] log_beta_VRS_hosp;
  log_beta_VRS_hosp = mean_log_beta_VRS_hosp + sdlogBetaVRSHosp * z_log_beta_VRS_hosp;

  vector[nAge+nWeek-1] pBeyfortus;
  pBeyfortus=inv_logit(logit_pBeyfortus);

  real probHOSP;

  // this season
    for (k in 1:nHopital) {
      num = rep_vector(0.0, nScenario);
      numHOSP = rep_vector(0.0, nScenario);
      denum=0;
      denumHOSP=0;
      for (j in 1:nAge) {
        for (i in 1:nWeek) {
           // duree 
        int d;
        if (j <= i + offsetIE) {
          d=j; //duree egale age si ne apres septembre
        } else {
          d=i+offsetIE;
        }
        if (i==1) {
            if (j==1) {
              pVRS[k,i,j]=inv_logit(log_beta_VRS_hosp[k]);
              pHOSP[k,i,j] = (inv_logit(delta2023Hopital[k]+logit_nu)*pBeyfortus[i-j+nAge]*(1.0-pVRS[k,i,j] * IE[d]) +
                                                         inv_logit(delta2023Hopital[k])*(1.0-pBeyfortus[i-j+nAge]))/
                                                         (1.0-pVRS[k,i,j]*pBeyfortus[i-j+nAge] * IE[d]);
              for (l in 1:nScenario) {
                NCON[l,k,i,j] = exp(mu2023Hopital[k]) *(1.0-pVRS[k,i,j] * pBeyScenario[l] * IE[d]);
                probHOSP =  (inv_logit(delta2023Hopital[k]+logit_nu)* pBeyScenario[l] *(1.0-pVRS[k,i,j] * IE[d]) +
                                                         inv_logit(delta2023Hopital[k])*(1.0-pBeyScenario[l]))/
                                                         (1.0-pVRS[k,i,j]*pBeyScenario[l] * IE[d]);
                NHOSPCON[l,k,i,j] = NCON[l,k,i,j] * probHOSP;                
              }
              NOBS[k,i,j] = NCON[1,k,i,j]*(1.0-pVRS[k,i,j]*pBeyfortus[i-j+nAge] * IE[d]);
              NHOSPOBS[k,i,j] = NOBS[k,i,j] *pHOSP[k,i,j];
            } else {
              pVRS[k,i,j]=inv_logit( log_beta_VRS_hosp[k] + log_beta_VRS_age[j-1]);
              pHOSP[k,i,j]= (inv_logit(delta2023Hopital[k]+deltaHopitalAge[k,j-1]+logit_nu)*pBeyfortus[i-j+nAge]*(1.0-pVRS[k,i,j] * IE[d]) +
                                                         inv_logit(delta2023Hopital[k]+deltaHopitalAge[k,j-1])*(1.0-pBeyfortus[i-j+nAge]))/
                                                         (1.0-pVRS[k,i,j]*pBeyfortus[i-j+nAge] * IE[d]);
              for (l in 1:nScenario) {
                NCON[l,k,i,j] = exp(mu2023Hopital[k]+muHopitalAge[k,j-1]) * (1.0-pVRS[k,i,j] * pBeyScenario[l] * IE[d]);
                probHOSP = (inv_logit(delta2023Hopital[k]+deltaHopitalAge[k,j-1]+logit_nu)*pBeyScenario[l]*(1.0-pVRS[k,i,j] * IE[d]) +
                                                           inv_logit(delta2023Hopital[k]+deltaHopitalAge[k,j-1])*(1.0-pBeyScenario[l]))/
                                                         (1.0-pVRS[k,i,j]*pBeyScenario[l] * IE[d]);
                NHOSPCON[l,k,i,j] = NCON[l,k,i,j] * probHOSP;
              }
                NOBS[k,i,j] = NCON[1,k,i,j]*(1.0-pVRS[k,i,j]*pBeyfortus[i-j+nAge] * IE[d]);
                NHOSPOBS[k,i,j] = NOBS[k,i,j] *pHOSP[k,i,j];
            }
          } else {
            if (j==1)  {
              pVRS[k,i,j]=inv_logit( log_beta_VRS_hosp[k] + log_beta_VRS_week[i-1]);
              pHOSP[k,i,j] = (inv_logit(delta2023Hopital[k]+deltaTime[i-1]+logit_nu)*pBeyfortus[i-j+nAge]*(1.0-pVRS[k,i,j] * IE[d]) +
                                                         inv_logit(delta2023Hopital[k]+deltaTime[i-1])*(1.0-pBeyfortus[i-j+nAge]))/
                                                         (1.0-pVRS[k,i,j]*pBeyfortus[i-j+nAge] * IE[d]);
              for (l in 1:nScenario) {
                NCON[l,k,i,j] = exp(mu2023Hopital[k]+ muTime[i-1]) * (1.0-pVRS[k,i,j] * pBeyScenario[l] * IE[d]);
                probHOSP=(inv_logit(delta2023Hopital[k]+logit_nu)*pBeyScenario[l]*(1.0-pVRS[k,i,j] * IE[d]) +
                                                         inv_logit(delta2023Hopital[k])*(1.0-pBeyScenario[l]))/
                                                         (1.0-pVRS[k,i,j]*pBeyScenario[l] * IE[d]);
                NHOSPCON[l,k,i,j] = NCON[l,k,i,j] * probHOSP;
              }
              NOBS[k,i,j] = NCON[1,k,i,j]*(1.0-pVRS[k,i,j]*pBeyfortus[i-j+nAge] * IE[d]);
              NHOSPOBS[k,i,j] = NOBS[k,i,j] * pHOSP[k,i,j];
            } else {
              pVRS[k,i,j]=inv_logit( log_beta_VRS_hosp[k] + log_beta_VRS_week[i-1] + log_beta_VRS_age[j-1]);
              pHOSP[k,i,j] = (inv_logit(delta2023Hopital[k]+deltaHopitalAge[k,j-1]+deltaTime[i-1]+logit_nu)*pBeyfortus[i-j+nAge]*(1.0-pVRS[k,i,j] * IE[d]) +
                                                         inv_logit(delta2023Hopital[k]+deltaHopitalAge[k,j-1]+deltaTime[i-1])*(1.0-pBeyfortus[i-j+nAge]))/
                                                         (1.0-pVRS[k,i,j]*pBeyfortus[i-j+nAge] * IE[d]);
              for (l in 1:nScenario) {
                NCON[l,k,i,j] = exp(mu2023Hopital[k]+ muHopitalAge[k,j-1]+muTime[i-1]) * (1.0-pVRS[k,i,j] * pBeyScenario[l] * IE[d]);
                probHOSP =  (inv_logit(delta2023Hopital[k]+deltaHopitalAge[k,j-1]+deltaTime[i-1]+logit_nu)*pBeyScenario[l]*(1.0-pVRS[k,i,j] * IE[d]) +
                                                         inv_logit(delta2023Hopital[k]+deltaHopitalAge[k,j-1]+deltaTime[i-1])*(1.0-pBeyScenario[l]))/
                                                         (1.0-pVRS[k,i,j]*pBeyScenario[l] * IE[d]);
                NHOSPCON[l,k,i,j] = NCON[l,k,i,j] * probHOSP;
              }
              NOBS[k,i,j] = NCON[1,k,i,j]*(1.0-pVRS[k,i,j]*pBeyfortus[i-j+nAge] * IE[d]);
              NHOSPOBS[k,i,j] = NOBS[k,i,j] * pHOSP[k,i,j];

            }
          }

          num[1]=num[1]+NCON[1,k,i,j]-NOBS[k,i,j];
          numHOSP[1] = numHOSP[1] + NHOSPCON[1,k,i,j] - NHOSPOBS[k,i,j];
          for (l in 2:nScenario) {
            num[l]=num[l]+NCON[1,k,i,j]-NCON[l,k,i,j];
            numHOSP[l] = numHOSP[l] + NHOSPCON[1,k,i,j] - NHOSPCON[l,k,i,j];
          }
          denum=denum+NCON[1,k,i,j];
          denumHOSP=denumHOSP+NHOSPCON[1,k,i,j];
        }
      }
      for (l in 1:nScenario){
        impactBRO[l,k]=num[l]/denum;
        impactHOSP[l,k]=numHOSP[l]/denumHOSP;
      }
    }
        
    // past Seasons
  for (l in 1:nSeason) {
    for (k in 1:nHopital) {
      for (j in 1:nAge) {
        for (i in 1:nWeek) {
          if (i==1) {
            if (j==1) { 
              NPASTBRO[l,k,i,j] = exp(muSeasonHopital[l,k]);
              NPASTHOSP[l,k,i,j]=NPASTBRO[l,k,i,j] * inv_logit(deltaSeasonHopital[l,k]);
            } else {
              NPASTBRO[l,k,i,j] = exp(muSeasonHopital[l,k] + muHopitalAge[k,j-1]);
              NPASTHOSP[l,k,i,j]=NPASTBRO[l,k,i,j] * inv_logit(deltaSeasonHopital[l,k]+ deltaHopitalAge[k,j-1]);
            }
          } else {
            if (j==1) {
              NPASTBRO[l,k,i,j] = exp(muSeasonHopital[l,k] + muSeasonTime[l,i-1]);
              NPASTHOSP[l,k,i,j]=NPASTBRO[l,k,i,j] * inv_logit(deltaSeasonHopital[l,k]+ deltaSeasonTime[l,i-1]);
            } else {
              NPASTBRO[l,k,i,j] = exp(muSeasonHopital[l,k] + muSeasonTime[l,i-1] + muHopitalAge[k,j-1]);
              NPASTHOSP[l,k,i,j]=NPASTBRO[l,k,i,j] * inv_logit(deltaSeasonHopital[l,k]+ deltaSeasonTime[l,i-1] + deltaHopitalAge[k,j-1]);
            }
          }
        }
      }
    }
  }
      

}


