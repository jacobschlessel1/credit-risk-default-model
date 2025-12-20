# Credit Risk & Loan Default Modeling

End-to-end credit risk modeling project using LendingClub data.

**Tools**
- Python
- PyTorch (later)
- AWS S3
- pandas / scikit-learn

**Goal**
Predict probability of default (PD) at loan origination using historical LendingClub data.

Raw data is stored securely in AWS S3 and loaded programmatically.

This project will model the following three systems:

- Probability of Hardship (PH)
- Probability of Default (PD)
- Loss Given Default (LGD)

**Variable Definitions** 

## Hardship & Payment Plan Variables

- `orig_projected_additional_accrued_interest`: Projected additional interest expected to accrue during a hardship period at the time the hardship plan was initiated.
- `hardship_payoff_balance_amount`: Remaining loan balance at the time a hardship payoff plan was established.
- `payment_plan_start_date`: Date on which a borrower’s hardship payment plan began.
- `hardship_last_payment_amount`: Amount of the most recent payment made under a hardship plan.
- `hardship_status`: Current status of the borrower’s hardship plan (e.g., active, completed).
- `hardship_start_date`: Date when the hardship plan began.
- `deferral_term`: Length of the payment deferral period under a hardship plan, in months.
- `hardship_amount`: Reduced payment amount agreed upon during the hardship period.
- `hardship_dpd`: Days past due at the time the hardship plan was initiated.
- `hardship_loan_status`: Loan status at the time the borrower entered the hardship program.
- `hardship_length`: Duration of the hardship plan, in months.
- `hardship_end_date`: Date when the hardship plan ended or is scheduled to end.
- `hardship_reason`: Borrower-reported reason for entering the hardship program (e.g., job loss).
- `hardship_type`: Type of hardship assistance provided (e.g., payment deferral, reduced payment).

## Debt Settlement Variables

- `settlement_status`: Current status of a debt settlement agreement.
- `debt_settlement_flag_date`: Date when the loan was flagged as being in debt settlement.
- `settlement_date`: Date the debt settlement agreement was finalized.
- `settlement_amount`: Total amount agreed upon to settle the debt.
- `settlement_term`: Duration of the debt settlement agreement, in months.
- `settlement_percentage`: Percentage of the outstanding balance that was agreed to be paid under settlement.

## Secondary Applicant / Joint Loan Variables

- `sec_app_mths_since_last_major_derog`: Months since the secondary applicant’s most recent major derogatory credit event.
- `sec_app_revol_util`: Revolving credit utilization rate of the secondary applicant.
- `revol_bal_joint`: Total revolving credit balance for joint applicants.
- `sec_app_open_acc`: Number of open credit accounts for the secondary applicant.
- `sec_app_num_rev_accts`: Number of revolving credit accounts for the secondary applicant.
- `sec_app_inq_last_6mths`: Number of credit inquiries for the secondary applicant in the last 6 months.
- `sec_app_mort_acc`: Number of mortgage accounts for the secondary applicant.
- `sec_app_open_act_il`: Number of currently active installment loans for the secondary applicant.
- `sec_app_fico_range_low`: Lower bound of the secondary applicant’s FICO score range at origination.
- `sec_app_fico_range_high`: Upper bound of the secondary applicant’s FICO score range at origination.
- `sec_app_chargeoff_within_12_mths`: Number of charge-offs on the secondary applicant’s credit report in the past 12 months.
- `sec_app_collections_12_mths_ex_med`: Number of non-medical collection accounts for the secondary applicant in the past 12 months.
- `sec_app_earliest_cr_line`: Date of the secondary applicant’s earliest reported credit line.
- `verification_status_joint`: Verification status of income for joint applicants.
- `dti_joint`: Debt-to-income ratio calculated using combined income for joint applicants.
- `annual_inc_joint`: Combined annual income of joint applicants.

## Borrower Description Field

- `desc`: Free-text description provided by the borrower explaining the purpose of the loan.

## Delinquency & Credit History Timing Variables

- `mths_since_last_record`: Months since the borrower’s most recent public record (e.g., bankruptcy).
- `mths_since_recent_bc_dlq`: Months since the borrower’s most recent bankcard delinquency.
- `mths_since_last_major_derog`: Months since the borrower’s most recent major derogatory credit event.
- `mths_since_recent_revol_delinq`: Months since the borrower’s most recent revolving account delinquency.
- `mths_since_last_delinq`: Months since the borrower’s most recent delinquency of any type.

## Credit Utilization & Recent Credit Activity Variables

- `il_util`: Utilization rate on installment loans.
- `mths_since_rcnt_il`: Months since the most recent installment loan was opened.
- `all_util`: Aggregate utilization rate across all credit accounts.
- `open_acc_6m`: Number of open credit accounts in the past 6 months.
- `inq_last_12m`: Number of credit inquiries in the past 12 months.
- `total_cu_tl`: Number of currently utilized trade lines.
- `open_il_24m`: Number of installment loans opened in the past 24 months.
- `open_il_12m`: Number of installment loans opened in the past 12 months.
- `open_act_il`: Number of currently active installment loans.
- `max_bal_bc`: Maximum current balance across all bankcard accounts.
- `inq_fi`: Number of finance-related credit inquiries.
- `total_bal_il`: Total outstanding balance on installment loans.
- `open_rv_24m`: Number of revolving accounts opened in the past 24 months.
- `open_rv_12m`: Number of revolving accounts opened in the past 12 months.

## Employment & Income Variables

- `emp_title`: Borrower’s job title as reported at loan application.
- `emp_length`: Borrower’s length of employment, reported in years.
- `annual_inc`: Borrower’s self-reported annual income at loan origination.

## Credit Account Counts & Balances

- `num_tl_120dpd_2m`: Number of accounts that were 120 or more days past due in the last 2 months.
- `mo_sin_old_il_acct`: Months since the borrower’s oldest installment loan was opened.
- `bc_util`: Utilization rate of revolving bankcard credit.
- `percent_bc_gt_75`: Percentage of bankcard accounts with utilization greater than 75%.
- `bc_open_to_buy`: Available credit on revolving bankcard accounts.
- `mths_since_recent_bc`: Months since the most recent bankcard account was opened.
- `pct_tl_nvr_dlq`: Percentage of accounts that have never been delinquent.
- `avg_cur_bal`: Average current balance across all credit accounts.
- `mo_sin_rcnt_rev_tl_op`: Months since the most recent revolving account was opened.
- `num_rev_accts`: Number of revolving credit accounts.
- `mo_sin_old_rev_tl_op`: Months since the oldest revolving account was opened.
- `num_op_rev_tl`: Number of currently open revolving accounts.
- `tot_coll_amt`: Total amount currently in collections.
- `num_actv_rev_tl`: Number of active revolving accounts.
- `total_rev_hi_lim`: Total revolving credit limit.
- `tot_cur_bal`: Total current balance across all credit accounts.
- `num_tl_30dpd`: Number of accounts currently 30 days past due.
- `num_actv_bc_tl`: Number of active bankcard accounts.
- `num_accts_ever_120_pd`: Number of accounts that have ever been 120+ days past due.
- `num_rev_tl_bal_gt_0`: Number of revolving accounts with a balance greater than zero.
- `mo_sin_rcnt_tl`: Months since the most recent credit account was opened.
- `num_tl_90g_dpd_24m`: Number of accounts 90+ days past due in the past 24 months.
- `tot_hi_cred_lim`: Total high credit limit across all accounts.
- `total_il_high_credit_limit`: Total high credit limit on installment loans.
- `num_il_tl`: Number of installment loan accounts.
- `num_tl_op_past_12m`: Number of accounts opened in the past 12 months.
- `num_bc_tl`: Number of bankcard accounts.
- `num_bc_sats`: Number of satisfactory bankcard accounts.
- `num_sats`: Number of satisfactory credit accounts.
- `total_bal_ex_mort`: Total balance on all non-mortgage credit accounts.
- `mort_acc`: Number of mortgage accounts.
- `acc_open_past_24mths`: Number of credit accounts opened in the past 24 months.
- `total_bc_limit`: Total credit limit across all bankcard accounts.

## Payment & Outcome Variables (Post-Origination)

- `last_pymnt_d`: Date of the most recent loan payment.
- `next_pymnt_d`: Scheduled date of the next loan payment.
- `last_pymnt_amnt`: Amount of the most recent payment.
- `recoveries`: Amount recovered after a charge-off.
- `collection_recovery_fee`: Fees associated with recovery after collections.
- `total_rec_prncp`: Total principal received to date.
- `total_rec_int`: Total interest received to date.
- `total_rec_late_fee`: Total late fees received.

## Administrative & Origination Variables

- `zip_code`: Borrower’s ZIP code (partially masked).
- `loan_amnt`: Amount of the loan requested by the borrower.
- `funded_amnt`: Amount funded by LendingClub.
- `funded_amnt_inv`: Amount funded by investors.
- `term`: Loan repayment term in months.
- `purpose`: Stated purpose of the loan.
- `issue_d`: Month and year the loan was issued.
- `loan_status`: Current status of the loan (e.g., fully paid, charged off).
