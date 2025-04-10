##################################### Load modules ################################################
using LinearAlgebra
using DataFrames
using CSV
using Random
using Statistics
using Distributions

#################################### Define functions ################################################
# function for 3D stress transformation
function StressTransform3D(Pf, SH, Sh, Sv, phi, theta)
    # Set up principal stress matrix
    s = [SH-Pf 0 0; 0 Sh-Pf 0; 0 0 Sv-Pf]
    
    # Pre-calculate trigonometric values
    cos_phi = cosd(phi)
    sin_phi = sind(phi)
    cos_theta = cosd(theta)
    sin_theta = sind(theta)
    
    # Perform stress transformation
    # Rotation matrix around Z-axis (vertical stress direction)
    Rz = [cos_phi -sin_phi 0; sin_phi cos_phi 0; 0 0 1]
    # Rotation matrix around X-axis (maximum horizontal stress direction)
    Rx = [1 0 0; 0 cos_theta -sin_theta; 0 sin_theta cos_theta]
    
    # Compute rotation matrix using rotation components in reverse order
    R = Rx * Rz
    
    # Perform rotation
    result = R * s * transpose(R)
    
    # Retrieve stresses from matrices
    tau1 = result[3, 1]  # Shear stress component in the strike direction
    tau2 = result[3, 2]  # Shear stress component in the dip direction
    tau = sqrt(result[3, 1]^2 + result[3, 2]^2)  # Shear stress
    sigma = result[3, 3]  # Normal stress
    
    return sigma, tau
end

# Function to generate samples and calculate moment and magnitude
function calculate_seismic_moment_magnitude(fault_slip_mean, fault_area, stress_drop, n_samples, rand_seed)
    # Generate random samples for A (ruptured area) and tau (stress drop)
    Random.seed!(rand_seed)
    A_samples = rand(Uniform(fault_slip_mean * fault_area, fault_area), n_samples)
    tau_samples = rand(Uniform(0.1 * stress_drop, stress_drop), n_samples)
    
    # Calculate seismic moment M0
    M0_samples = tau_samples .* (A_samples .^ 1.5)
    
    # Calculate max and mean for M0
    M0_max = maximum(M0_samples)
    M0_mean = mean(M0_samples)
    
    # Calculate Mw
    Mw_max = (2 / 3) * (log10(M0_max) - 9.1)
    Mw_mean = (2 / 3) * (log10(M0_mean) - 9.1)
    
    return M0_max, M0_mean, Mw_max, Mw_mean
end

# Main function
function calculate_seismic_hazard(geom_params, fault_info, df_temp_initial, df_temp_press, rand_seed, saveData::Bool)
    #################################### Load geomechanical data ################################################
    SH_grad = geom_params["SH_grad"]
    Sh_grad = geom_params["Sh_grad"]
    Sv_grad = geom_params["Sv_grad"]
    SH_azi = geom_params["SH_azi"]
    mu = geom_params["mu"]
    E = geom_params["E"]
    PR = geom_params["PR"]
    alpha = geom_params["alpha"]
    cohesion = geom_params["cohesion"]
    
    #################################### Load fault, pressure, and temperature data ################################################
    # load fault info
    df_byFault = DataFrame(fault_info)
    # rename the temp/press dataframe
    df_byCell = df_temp_press
    # add initial temp column
    df_byCell = leftjoin(df_byCell, df_temp_initial[:, [:i, :j, :k, :temp_step1]], on = [:i, :j, :k])
    # add 'strike' and 'dip' columns, use innerjoin to remove irrelavent faults
    df_byCell = innerjoin(df_byCell, df_byFault[:, [:fault_id, :strike_deg, :dip_deg]], on=:fault_id)
    
    #################################### Calculate fault slip indicator for each fault cell ################################################
    df_byCell[!, "SH_MPa"] = df_byCell[:, "z"] / 1000 * SH_grad
    df_byCell[!, "Sh_MPa"] = df_byCell[:, "z"] / 1000 * Sh_grad
    df_byCell[!, "Sv_MPa"] = df_byCell[:, "z"] / 1000 * Sv_grad
    df_byCell[!, "SH_azi_deg"] = fill(SH_azi, nrow(df_byCell))
    df_byCell[!, "mu"] = fill(mu, nrow(df_byCell))
    df_byCell[!, "E_GPa"] = fill(E, nrow(df_byCell))
    df_byCell[!, "PR"] = fill(PR, nrow(df_byCell))
    df_byCell[!, "alpha_/C"] = fill(alpha, nrow(df_byCell))
    df_byCell[!, "cohesion_MPa"] = fill(cohesion, nrow(df_byCell))
    # # df['deltaT_C'] = df['temp_step1'] - df['temp']
    df_byCell[!, "deltaT_C"] = 90 .- df_byCell[:, :temp]  # CHANGE later on

    # # generate data for verification
    # CSV.write("verification/v6_Julia.csv",df_byCell)
    
    # Compute rotation angles
    df_byCell[!, "phi_deg"] = df_byCell[:, "SH_azi_deg"] .- df_byCell[:, "strike_deg"]
    df_byCell[df_byCell[:, "phi_deg"] .< 0, "phi_deg"] .+= 360
    df_byCell[!, "theta_deg"] = df_byCell[:, "dip_deg"]
    df_byCell[df_byCell[:, "theta_deg"] .> 90, "phi_deg"] .+= 180
    df_byCell[!, "theta_deg"] = [x > 90 ? 180 - x : x for x in df_byCell[:, "theta_deg"]]
    df_byCell[df_byCell[:, "phi_deg"] .> 360, "phi_deg"] .-= 360
    
    # 3D stress transformation
    results = [StressTransform3D(
        row[:pressure] / 10,
        row[:SH_MPa],
        row[:Sh_MPa],
        row[:Sv_MPa],
        row[:phi_deg],
        row[:theta_deg] 
    ) for row in eachrow(df_byCell)]
   
    # Creating two separate arrays for sigma_MPa and tau_MPa
    sigma_MPa = [result[1] for result in results]
    tau_MPa = [result[2] for result in results]

    # Adding the results back into the DataFrame as new columns
    df_byCell[!, "sigma_MPa"] = sigma_MPa
    df_byCell[!, "tau_MPa"] = tau_MPa
    
    # Calculate thermally induced normal stress
    df_byCell[:, "sigmaThermal_MPa"] = 1000 .* df_byCell[:, "E_GPa"] .* df_byCell[:, "alpha_/C"] .* df_byCell[:, "deltaT_C"] ./ (1 .- 2 .* df_byCell[:, "PR"])
    
    # Calculate fault slip indicator
    df_byCell[!, "fault_slip"] = [
    ((tau - cohesion) / (sigma - sigmaThermal) < mu) ? 0 : 1 
    for (sigma, tau, mu, cohesion, sigmaThermal) 
    in zip(df_byCell[:, "sigma_MPa"], df_byCell[:, "tau_MPa"], df_byCell[:, "mu"], df_byCell[:, "cohesion_MPa"], df_byCell[:, "sigmaThermal_MPa"])
    ]

    # # for verification
    # CSV.write("verification/v6_Julia_results.csv", df_byCell)

    #################################### Calculate summary statistics for each fault ################################################
    # Group by 'fault_id' and calculate summary statistics
    df_grouped = groupby(df_byCell, :fault_id)

    # Define aggregation functions and their corresponding new column names
    agg_functions = [
        :fault_slip => sum => :fault_slip_sum,
        :fault_slip => length => :fault_slip_count,
        :fault_slip => mean => :fault_slip_mean,
        :tau_MPa => mean => :tau_MPa_mean
    ]

    # Apply the aggregation functions and combine the results
    df_summary = combine(df_grouped, agg_functions...)
    
    # Add fault total area
    df_summary = leftjoin(df_summary, df_byFault[:, [:fault_id, :fault_total_area_m2]], on=:fault_id)
    
    ################################### Estimate seismic moment to represent seismic hazard ##########################################
    slipped_threshold = 0
    select_cols = [:fault_id, :fault_slip_mean, :fault_total_area_m2, :tau_MPa_mean]
    df_M0 = df_summary[(df_summary[:, "fault_slip_mean"] .> slipped_threshold), select_cols]
    
    n_samples = 1000
    results_m0 = [calculate_seismic_moment_magnitude(
        row[:fault_slip_mean], 
        row[:fault_total_area_m2], 
        row[:tau_MPa_mean], 
        n_samples,
        rand_seed
    ) for row in eachrow(df_M0)]

    # Unzip the results into separate vectors
    M0_max, M0_mean, Mw_max, Mw_mean = map(x -> [r[x] for r in results_m0], 1:4)

    # Add new columns to the DataFrame
    df_M0[!, "M0_max"] = M0_max
    df_M0[!, "M0_mean"] = M0_mean
    df_M0[!, "Mw_max"] = Mw_max
    df_M0[!, "Mw_mean"] = Mw_mean

    # Compute total seismic moment to represent seismic hazard
    SH = sum(df_M0[:, "M0_mean"])
    
    # Save data
    if saveData
        CSV.write("results_v6_faultAll.csv", df_byCell)
        CSV.write("results_v6_faultAll_summary.csv", df_summary)
        CSV.write("results_v6_faultAll_M0.csv", df_M0)
        println("df_byCell, df_summary and df_MO have been saved as csv files.")
    else
        println("Skipping saving data.")
    end
    
    return SH, df_M0
end
