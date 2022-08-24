clc; 
H_ex = load('../data/H_16x64_MIMO_CDL_A_ULA_test.mat').hest;
Nt = 64;
Nr = 16;
N_s = 16;
N_p_vec = [25];
SNR_vec = -10:5:15;
nrepeat = 10;
ntest = 20;
j = sqrt(-1);
qpsk_constellation = (1/sqrt(2))*[1+1j,1-1j,-1+1j,-1-1j];
identity = eye(Nr);
A1 = dftmtx(Nt)/sqrt(Nt); 
A2 = dftmtx(Nr)/sqrt(Nr); 
nmse_arr = zeros(length(SNR_vec),1);
Nbit_r = 2;
Nbit_t = 6;
Sparsity=5;            % Hyperparameter for GAMP - initial value
Ncs=Nt*Nr;
BGmean=0;
sparseRat = Sparsity/Ncs;
BGvar=0.05;
wvar = 1;                       % Scale measurements according to SNR
map = false;
hybrid = 1;
capacity_1bitQ_CDL_A = zeros(length(SNR_vec),1);

for b = 1:length(N_p_vec)
    N_p = N_p_vec(b);
    for a = 1:nrepeat
        disp(a);
        if hybrid == 1
            pilot_sequence_ind = randi([1,4],N_s,N_p);
            pilot_sequence = qpsk_constellation(pilot_sequence_ind);
            precoder_training = training_precoder(Nt,N_s,Nbit_t);
            W = training_combiner(Nr,N_s,Nbit_r); 
            A = kron(transpose(precoder_training*pilot_sequence),W);
            A_H = A';
            A_tx = kron(transpose(precoder_training*pilot_sequence),identity);
            A_sp = kron(transpose((A1')*precoder_training*pilot_sequence),W*A2);
        else
            pilot_sequence_ind = randi([1,4],Nt,N_p);
            pilot_sequence = qpsk_constellation(pilot_sequence_ind);
            A_tx = kron(transpose(pilot_sequence),identity);
            A = A_tx;
            A_H = A_tx';
            A_sp = kron(transpose((A1')*pilot_sequence),A2);
        end
        for i = 1:ntest
            vec_H_single = reshape(H_ex(:,:,i),Nr*Nt,1);
            signal = A_tx*vec_H_single;
            E_s = signal.*conj(signal);
            noise = (randn(Nr*N_p,1)+j*randn(Nr*N_p,1));
            for k = 1:length(SNR_vec)
                SNRdB = SNR_vec(k);           % SNR in dB.
                std_dev = (1/(10^(SNRdB/20)))*sqrt(E_s);
                if hybrid == 1
                    noise_matrix = (1/sqrt(2))*W*reshape(std_dev.*noise,[Nr,N_p]);
                else
                    noise_matrix = (1/sqrt(2))*reshape(std_dev.*noise,[Nr,N_p]);
                end
                y = A*vec_H_single + reshape(noise_matrix,[Nr*N_p,1]);
                r_bit_real = sign(real(y));
                r_bit_imag = sign(imag(y));
                r_train = r_bit_real + 1j * r_bit_imag;  
                inputEst = CAwgnEstimIn(BGmean, BGvar, map);
                inputEst = SparseScaEstim(inputEst,sparseRat,0);
                outputEst = CProbitEstimOut(r_train,0,1,false);
                opt = GampOpt();
                opt.legacyOut=false;
                [estFin,optFin,estHist] = gampEst(inputEst, outputEst, A_sp, opt);
                xhat = estFin.xhat;
                H_est = reshape(A2*reshape(xhat,Nr,Nt)*A1',Nr*Nt,1);

                % Display the MSE
                alph = sum(conj(H_est) .* vec_H_single) / sum(conj(H_est) .* H_est);
                nmseGAMP = (norm(vec_H_single-alph*H_est)/norm(vec_H_single))^2;
                nmse_arr(k,1) = nmse_arr(k,1) + nmseGAMP;

                %Capacity Computation
                H = reshape(alph*H_est,Nr,Nt);
                [U,S,V] = svd(H);
                H_true = reshape(H_ex(:,:,i),Nr,Nt);
                gain = abs(U(:,1)'*H*V(:,1));
                C = log2(1 + (10^(SNRdB/10))*gain*gain);
                capacity_1bitQ_CDL_A(k,1) = capacity_1bitQ_CDL_A(k,1) + C;
            end
        end
    end
end
nmse_arr = nmse_arr/(nrepeat*ntest);