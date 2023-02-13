% E = keplersolve(M,e)
function E = keplersolve(M,e)
MAXITER = 100; % in case the loop does not converge
TOL = 1e-11; % tolerance for convergence
En = M; % first guess
fn = En - e*sin(En) - M; % f(E) that we want to be zero
iter = 1; % count the iterations
while ( abs(fn) > TOL )
En = En - fn/(1-e*cos(En));
fn = En - e*sin(En) - M;
iter = iter+1;
if (iter>=MAXITER)
error('Did not converge');
exit
end
end
E = En;