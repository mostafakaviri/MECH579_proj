# 25.12.04
# 1. gradient 벡터 구하는 함수를 numpy 연산 함수가 아닌 torch 연산 함수를 인자로 받아 AD를 수행하는 함수로 수정(grad_ad)
# 2. 이에 따라 모든 solver 및 내부 함수는 torch 연산 함수(*_torch)를 추가 인자로 받는다. 따라서 사용자가 torch 연산 함수를 solver 인자에 입력해주어야 한다.
# 3. 그리고 alm, alm4sqp 내부 convergence criteria에 쓰이는 ∇L_A -> ∇L로 수정하였음 

############################################# Numerical Optimization 모듈(완성본 모음) #############################################
import numpy as np
import io
import contextlib
import torch

# --------------------------------------------------------------------------Derivative/Hessian Caculation----------------------------------------------------
### Central Difference Method based Gradient - Scalar func / n-dim point x
# scipy.differentiate.derivative(f, x, ...) 함수 사용 가능
def grad_ad(f_torch, x):
    x = torch.tensor(x, requires_grad=True)
    fx = f_torch(x).backward()
    dfdx = x.grad.detach().numpy().astype(np.float64)
    if not np.isfinite(dfdx).all(): # Check finitude
        raise ValueError('At least one component of gradient is not finite !')
    return dfdx

# --------------------------------------------------------------------------Direction(p) search algorithms--------------------------------------------------------------
### Search direction using Steepest Descent Method - Scalar func / n-dim point x
def search_direction_stp_descent(f_torch, x):
    p = -grad_ad(f_torch, x)

    if not np.isfinite(p).all():
        raise ValueError("Non-finite number detected in search_direction_stp_descent (NaN or inf)")

    return p

### Search direction using Conjugate Gradient Method - Hestenes Stiefel(CGM-HS) Formula
# Nonlinear CG 중 Hestenes-Stiefel algorithm 사용 시 beta 계산에서 분모가 0 되면 폭주 가능.
# 따라서 Steepest descent랑 섞어서 그런 부분 방지해야 함.
# scipy.optimize.minimize(..., method='CG', ...) 함수 사용 가능
def search_direction_cg_hs(k, grad_old, grad_cur, p_old):
    if k == 0:
        p = -grad_cur
    else:
        num = (grad_cur - grad_old) @ grad_cur
        den = (grad_cur - grad_old) @ p_old
        if abs(den) < 1e-12 or np.isnan(den) or np.isnan(num): # 분모(den)가 0에 가까워지면 beta, p, x_new가 차례로 폭주하는 걸 방지하기 위해 이 경우 steepest descent method를 대신 사용.
            p = -grad_cur
        else:
            beta = num/den
            p = -grad_cur + beta*p_old
    
    if not np.isfinite(p).all():
        raise ValueError("Non-finite number detected in search_direction_cg_hs (NaN or inf)")

    return p

### Search direction using Conjugate Gradient Method - Fletcher Reeves(CGM-FR) Formula
# Nonlinear CG 중 Fletcher-Reeves algorithm 사용 시 beta 계산에서 분모가 0 될 일이 없기에 폭주 가능성 없음.
# 따라서 Steepest descent랑 섞어서 안 써도 됨. -> 수정 : grad_old가 0에 가까울 시 분모 0 될 수 있음.
# scipy.optimize.minimize(..., method='CG', ...) 함수 사용 가능
def search_direction_cg_fr(k, grad_old, grad_cur, p_old):
    if k == 0:
        p = -grad_cur
    else:
        num = grad_cur @ grad_cur
        den = grad_old @ grad_old

        ####### steepest descent 부분 무효(주석)처리 ########
        if abs(den) < 1e-12 or np.isnan(den) or np.isnan(num): # 분모(den)가 0에 가까워지면 beta, p, x_new가 차례로 폭주하는 걸 방지하기 위해 이 경우 steepest descent method를 대신 사용.
            p = -grad_cur
        else:
            beta = num/den
            p = -grad_cur + beta*p_old

        beta = num/den
        p = -grad_cur + beta*p_old

    if not np.isfinite(p).all():
        raise ValueError("Non-finite number detected in search_direction_cg_fr (NaN or inf)")
        
    return p

### Search direction using Quasi-Newton Method - BFGS(QNM-BFGS) Formula
# scipy.optimize.minimize(..., method='BFGS', ...) 함수 사용 가능
def search_direction_quasi_newton_bfgs(k, x_old, x_cur, grad_old, grad_cur, hessian_inv_aprx_old):
    dim_x = len(x_cur)
    if k == 0: # 첫 iteration의 근사 Hessian inverse는 I로 설정
        hessian_inv_aprx = np.eye(dim_x)
    else: # 2번째 이후 iteration부터의 근사 Hessian inverse부터는 BFGS로 구함
        dx = x_cur - x_old
        dg = grad_cur - grad_old
        dgdx = dg @ dx
        if abs(dgdx) < 1e-10:  # to avoid division by zero
            hessian_inv_aprx = np.eye(dim_x)
        else:
            I = np.eye(dim_x)
            rho = 1.0 / dgdx
            V = I - rho * (dx.reshape(-1, 1) @ dg.reshape(1, -1)) # 벡터로 행렬 생성 연산은 @ 연산 쓰지 말고 대신 np.outer(a, b) 함수 써라.
            hessian_inv_aprx = V @ hessian_inv_aprx_old @ V.T + rho * np.outer(dx, dx) # BFGS formula

    p = -hessian_inv_aprx @ grad_cur

    if not np.isfinite(p).all():
        raise ValueError("Non-finite number detected in search_direction_quasi_newton_bfgs (NaN or inf)")

    return p, hessian_inv_aprx

# ---------------------------------------------------------------------------Step length(α) search algorithms---------------------------------------------------------------
### Step length search A) Backtracking algorithm - Scalar func / n-dim point x / n-dim grad_x / n-dim search direction p / curruent iteration k
# backtracking 알고리즘, 더 포괄적으로 step size alpha를 찾는 line search algorithm은 반드시 함수가 명시적으로 주어져야 한다.
# alpha를 찾기 위해서는 every alpha_try에서 function evaluation을 거쳐야 하기 때문이다.
def backtracking(func, x, grad_x, p, k):
    c1 = 1e-4
    c2 = 0.5

    alpha_try = 1
    x_try = x + alpha_try*p

    i = 0
    
    print(f"func(x)={func(x)}, func(x_try)={func(x_try)}, grad·p={grad_x.T@p}")    

    while func(x_try) > (func(x) + c1*alpha_try*grad_x.T@p):
        i = i + 1
        alpha_try = c2*alpha_try
        x_try = x + alpha_try*p

    alpha = alpha_try

    print(f'alpha_{k}_{i} = {alpha}\n')

    return alpha

### Step length search B) Strong Wolfe's Conditions + Interpolation algorithm- Scalar func / n-dim point x / n-dim grad_x / n-dim search direction p / curruent iteration k
# interpol_alpha ⊂ bracketing_alpha ⊂ wolfe_strong_interpol
def interpol_alpha(f, f_torch, x_cur, p_cur, a, b):
    # bracketing_alpha에서 계속해서 업데이트되는 구간(alpha_lo, alpha_hi)에 대해, 양단 점을 기반으로 2(3)차 함수로 근사하여 alpha_new 찾는 함수
    # 이 alpha_new는 bracketing_alpha에서 구간을 새로 업데이트할지 말지, 업데이트한다면 어떤 방향으로 업데이트할 지 판단할 때 사용
    phi_a = f(x_cur + a*p_cur)
    phi_b = f(x_cur + b*p_cur)
    dphi_a = grad_ad(f_torch, x_cur + a*p_cur)@p_cur
    
    dem = 2*(phi_b - phi_a - dphi_a*(b - a)) # 2차 함수 근사식 기반 minimum point 계산식의 분모
    
    if (abs(dem) < 1e-12) | (not np.isfinite(dem)): # 분모 수치적으로 불안정하면
        alpha_min = .5*(a + b) # 그냥 구간의 중심점을 minimum point로 상정하고 return
        return alpha_min
    else: # 분모 수치적으로 안정하면
        num = dphi_a*(b - a) # 분자 계산해주고
        alpha_min = a - num/dem # 2차근사식 minimum point 계산식 기반 minimum point 구해준다.
        alpha_min = np.clip(alpha_min, a + 0.1*(b - a), b - 0.1*(b - a)) # 만약 minimum point가 너무 작은 값이면(유의미한 point 이동 못 이뤄냄) 구간의 10% 지점으로, 너무 큰 값(너무 많이 point 이동해도 문제)이면 구간의 90% 지점으로 치환한다.
        return alpha_min    

# bracketing_alpha ⊂ wolfe_strong_interpol
def bracketing_alpha(f, f_torch, x_cur, p_cur, c2_dphi0, phi_armijo, alpha_lo, alpha_hi): 
    # wolfe_strong_interpol에서 확정된 '큰' 골짜기 구간을 기반으로 alpha_optm 찾는 함수
    # 여기서 추가적으로 구간을 더 작게 업데이트할 수도 있음 -> '작은' 골짜기 구간
    phi_lo = f(x_cur+alpha_lo*p_cur)

    for _ in range(50):
        alpha_new = interpol_alpha(f, f_torch, x_cur, p_cur, alpha_lo, alpha_hi) # 다항식 보간함수로 골짜기 구간에서의 최소추정점 alpha_new 구함
        phi_new = f(x_cur + alpha_new*p_cur) # alpha_new에서의 함수값
        dphi_new = grad_ad(f_torch, x_cur + alpha_new*p_cur)@p_cur # alpha_new에서의 기울기

        if (phi_new > phi_armijo(alpha_new)) | (phi_new >= phi_lo): # alpha_new에서의 함수값이 오히려 증가했다
            alpha_hi = alpha_new # alpha_optm은 alpha_lo와 alpha_new 사이 존재 -> '작은' 골짜기 구간 [alpha_lo, alpha_new]로 업뎃
            # alpha_lo unchanged
        else:
            if abs(dphi_new) <= c2_dphi0: # alpha_new에서의 함수값이 감소했고 기울기까지 작다
                alpha_optm = alpha_new # alpha_new가 alpha_optm
                print(f'α* satisfying strong Wolfe\'s condition !')
                return alpha_optm
            elif dphi_new > 0: # alpha_new에서의 함수값이 감소했는데 기울기는 여전히 양수다
                alpha_hi = alpha_new # alpha_optm은 alpha_lo와 alpha_new 사이 존재 -> '작은' 골짜기 구간 [alpha_lo, alpha_new]로 업뎃
                # alpha_lo unchanged
            else: # alpha_new에서의 함수값이 감소했는데 기울기가 음수다
                alpha_lo = alpha_new # alpha_optm은 alpha_new와 alpha_hi 사이 존재 -> '작은' 골짜기 구간 [alpha_new, alpha_hi]로 업뎃
                # alpha_hi unchanged
        
        phi_lo = f(x_cur + alpha_lo*p_cur) # 업뎃된 alpha_lo에서의 함수값 계산
        
        if abs(alpha_hi - alpha_lo) < 1e-8: # 만약 구간이 충분히 줄어들었으면 그냥 탈출해서 구간의 절반지점을 alpha_optm으로 return
            break
    print(f'α* not satisfying strong Wolfe\'s condition !')
    return 0.5*(alpha_lo + alpha_hi)

# wolfe_strong_interpol - 확실한 '큰' 골짜기 구간을 찾아 거기서 alpha_optm을 찾는 함수
def wolfe_strong_interpol(f, f_torch, x_cur, f_cur, grad_cur, p_cur, c2):
    c1 = 1e-4 # Armijo 조건, Curvature 조건용 factors
    alpha_try_old, alpha_try = 0, 1 # Initial bracket of alpha

    phi0 = f_cur # Armijo 함수 생성용
    dphi0 = grad_cur@p_cur # Armijo 함수 생성용 및 Curvature 조건 비교용

    phi_armijo = lambda alpha : phi0 + c1*alpha*dphi0 # Armijo 람다 함수 정의

    for _ in range(50):
        x_try = x_cur + alpha_try*p_cur
        phi_try = f(x_try)
        dphi_try = grad_ad(f_torch, x_try)@p_cur

        phi_armijo_try = phi_armijo(alpha_try)

        x_try_old = x_cur + alpha_try_old*p_cur
        phi_try_old = f(x_try_old)

        if (phi_try > phi_armijo_try) | (phi_try > phi_try_old): # phi_try가 충분히 크다면 -> alpha_optm이 alpha_try_old와 alpha_try 사이 존재
            alpha_lo, alpha_hi = alpha_try_old, alpha_try
            alpha_optm = bracketing_alpha(f, f_torch, x_cur, p_cur, abs(c2*dphi0), phi_armijo, alpha_lo, alpha_hi) # bracketing 하고 interpolation iteration 돌려서 alpha_optm 뽑아내자
            return alpha_optm

        elif abs(dphi_try) <= -c2*dphi0: # phi_try가 충분히 작고 기울기까지 작다면
            alpha_optm = alpha_try # 그 점이 alph_optm이다
            print(f'α* satisfying strong Wolfe\'s condition !')
            return alpha_optm

        elif dphi_try >= 0: # phi_try가 충분히 작긴 한데 기울기가 양수라면 더 작은 phi 값을 가지는 alpha_optm이 alpha_try_old와 alpha_try 사이 존재
            alpha_lo, alpha_hi = alpha_try_old, alpha_try
            alpha_optm = bracketing_alpha(f, f_torch, x_cur, p_cur, abs(c2*dphi0), phi_armijo, alpha_lo, alpha_hi) # bracketing 하고 interpolation iteration 돌려서 alpha_optm 뽑아내자
            return alpha_optm

        else: # phi_try가 충분히 작긴 한데 기울기가 음수라면 더 작은 phi 값을 가지는 alpha_optm은 alpha_try보다 뒤의 구간에 존재 -> 구간 업데이트
            alpha_try_old = alpha_try # 다음 구간의 하한 = 현재 구간의 상한
            alpha_try = min(alpha_try * 2, 10.0) # 다음 구간의 상한 = 현재 구간 상한의 2배. 최대는 10으로 제한
    
    if not np.isfinite(alpha_try):
        alpha_try = 1e-3
    print(f'α* not satisfying strong Wolfe\'s condition !')
    return max(min(alpha_try, 1.0), 1e-6)

# Step length search algorithm for SQP
def steplength_merit(k, mu, f, ce, ce_torch, ci, ci_torch, x_k, p_k, grad_f_k, B_k, lmbda_k, nu_k):
    ### mu(penalty parameter for l1 merit function) 선정
    ## mu 구하는 버전 1 : 강의자료(Coursenotes)의 3.3.5의 마지막 formula
    print(f'Step length search using l1 merit function ... ')
    pBp = p_k @ (B_k @ p_k)
    sum_c = sum([abs(ce_j(x_k)) for ce_j in ce]) + sum([np.maximum(-ci_j(x_k), 0) for ci_j in ci]) # sum of the constraints violation
    max_lm = max(max(abs(lmbda_k)), max(abs(nu_k))) # maximum lagrange multiplier
    print(f'max_lm = {max_lm}')
    rho = 0.2 # for mu 
    sigma = 1 if pBp > 0 else 0 # for mu
    den = (1 - rho) * sum_c
    num = (grad_f_k @ p_k) + 0.5 * sigma * pBp
    if den == 0: # sum_c == 0 인 케이스: candidate이 무한대로 발산할테니 merit function 망가질 예정
        mu = max_lm # 그럼 그냥 최대 라그랑지 승수를 mu로 선정
    else:
        candidate = num / den
        if not np.isfinite(candidate): # 혹시나 candidate이 finite하지 않다면
            mu = max_lm # 그럼 그냥 최대 라그랑지 승수를 mu로 선정
        else:
            mu = max(max_lm, candidate) # candidate이 정상적인 숫자로 계산되었다면 최대 라그랑지 승수와 비교해서 더 큰 값을 mu로 선정

    # ## mu 구하는 버전 2 : Nocedal 책의 formula 18.40
    # if k == 0:
    #     rho = 0.2 # for mu 
    #     sigma = 1 if pBp > 0 else 0 # for mu
    #     mu = max(max_lm, ((grad_f_k @ p_k) + .5*sigma*pBp)/((1 - rho)*sum_c)) # penalty parameter for constraint terms in phi1
    # else:
    #     delta = 1.0
    #     mu = mu if (mu**-1 >= (max_lm + delta)) else (max_lm + 2*delta)**-1
    
    print(f'mu = {mu}')

    ### merit function φ1 정의
    phi1 = lambda alpha : f(x_k + alpha*p_k) + mu*(sum([abs(ce_j(x_k + alpha*p_k)) for ce_j in ce]) + sum([np.maximum(-ci_j(x_k + alpha*p_k), 0) for ci_j in ci])) # l1 merit function
    
    # αk=0에서의 기울기 dφ1/dα(αk=0) 값 계산
    sum_grad_ce = np.array([-grad_ad(ce_torch_j, x_k) if ce_j(x_k) < 0.0 else grad_ad(ce_torch_j, x_k) for ce_j, ce_torch_j in zip(ce, ce_torch)]).sum(axis=0)
    sum_grad_ci = np.array([-grad_ad(ci_torch_j, x_k) if ci_j(x_k) < 0.0 else np.zeros_like(x_k) for ci_j, ci_torch_j in zip(ci, ci_torch)]).sum(axis=0)
    Dphi1_0 = (grad_f_k + mu*(sum_grad_ce + sum_grad_ci)) @ p_k # 내가 직접 구한 dφ1/dα(αk=0)
    # Dphi1_0 = p_k @ grad_f_k - mu*sum_c # Nocedal Lemma 18.2 543pg 에 나와있는 dφ1/dα(αk=0) 식
    print(f'Dphi1_0 = {Dphi1_0}')

    if Dphi1_0 > 1e-4:
        print(f'grad_f_k @ p_k = {grad_f_k @ p_k}')
        print(f'sum_grad_ce*p_k = {sum_grad_ce @ p_k}')
        print(f'sum_grad_ci*p_k = {sum_grad_ci @ p_k}')
        print(f'mu = {mu}')
        print(f'sum_c = {sum_c}')
        raise ValueError(f"Dphi1_0 = {Dphi1_0} is not a descent direction. Something goes wrong !")
    
    alpha_k = 1.0 # initial alpha_try
    eta = .01 # armijo level ... 클수록 요구 감소조건이 약하고 작을수록 요구 감소조건이 큼.
    beta = .5 # decreasing factor for alpha_try
    phi1_0 = phi1(0) # phi1 value at alpha = 0
    print(f'phi1_0 = {phi1_0}')
    for _ in range(100):
        phi1_k = phi1(alpha_k)
        print(f'phi1_k = {phi1_k}')
        armijo_k = phi1_0 + eta*alpha_k*Dphi1_0
        if phi1_k < armijo_k: # armijo condition
            return alpha_k, mu
        else:
            alpha_k = beta*alpha_k
    print(f'alpha_k did not meet the armijo condition ... ')
    return alpha_k, mu

# ----------------------------------------------------------------------------Main Optimization Algorithm(Unconstrained)--------------------------------------------------------
### Steepest Descent Method
# 가장 느리지만 가장 수렴 가능성 높음
def stp_descent(f, f_torch, x0, tol):
    ### Check input data type
    if (not isinstance(x0, np.ndarray)) | (x0.ndim >= 2):
        raise ValueError('Please input 1D ndarray type !!')

    ### Check f(x0)
    f0 = f(x0)
    if not np.isfinite(f0).all():
        raise ValueError('Function value at x0 is not finite. Try another x0 !')

    ### Check ||∇f(x0)||
    grad0 = grad_ad(f_torch, x0)
    if np.linalg.norm(grad0) < tol: # Check optimality
        print(f'Since ||grad(x0)|| = {np.linalg.norm(grad0)} < {tol}, x0 : {x0} is optimum point !')
        list_x = [x0]; list_f = [f0]; list_grad = [grad0]
        return list_x, list_f, list_grad
    else:
        print(f'Since ||grad(x0)|| = {np.linalg.norm(grad0)} > {tol}, x0 : {x0} is not an optimum point. Optimization begins !')
        pass

    ### Initialization for searching iterations
    x_new = x0
    f_new = f0
    grad_new = grad0
    err = 1
    k = 0

    ############## 과제용 plot을 위한 log 담기 위한 list
    list_x = [x0]
    list_f = [f0]
    list_grad = [grad0]

    ############## Searching iterations
    ###### Update info of current point
    while err > tol:
        x_cur = x_new
        f_cur = f_new
        grad_cur = grad_new

        ###### Line search
        ### Search direction - by steepest gradient method
        p_cur = search_direction_stp_descent(f_torch, x_cur)
        # Search direction check
        if grad_cur@p_cur > 0: # search direction이 증가 방향이면 경고 내보내고 steepest descent direction으로 search direction 변경
            print(f'Warning : grad(x_k)·p_k={grad_cur@p_cur} > 0 : Search direction p_k would likely make function increase !')
            print(f'Warning : p_k would be replaced with steepest descent direction grad(x_k) : {-grad_cur} !')
            p_cur = search_direction_stp_descent(f_torch, x_cur)

        ### Step length - by Strong Wolfe + interpolation
        alpha_cur = wolfe_strong_interpol(f, f_torch, x_cur, f_cur, grad_cur, p_cur, 0.9)

        ##### x_new update
        k = k + 1
        x_new = x_cur + alpha_cur*p_cur
        f_new = f(x_new)
        grad_new = grad_ad(f_torch, x_new)
        err = np.linalg.norm(grad_new)
        list_x.append(x_new)
        list_f.append(f_new)
        list_grad.append(grad_new)
        print(f'x_{k} : {x_new}')
        print(f'f_{k} : {f_new}')
        print(f'norm(grad(x_{k})) : {err}')
        print(f'recent alpha : {alpha_cur}')
        print(f'recent p : {p_cur}')
        print()

    print(f'Optimization converges -> Iteration : {k} / x* : {x_new} / f(x*) : {f(x_new)} / norm(grad(x*)) : {np.linalg.norm(grad_new)} ')
    return list_x, list_f, list_grad

### Congugate Gradient Method - Hestenes Stiefel(CGM-HS)
# 적당히 빠르고 수렴도 꽤 잘 됨
def cg_hs(f, f_torch, x0, tol):
    ### Check input data type
    if (not isinstance(x0, np.ndarray)) | (x0.ndim >= 2):
        raise ValueError('Please input 1D ndarray type !!')

    ### Check f(x0)
    f0 = f(x0)
    if not np.isfinite(f0).all():
        raise ValueError('Function value at x0 is not finite. Try another x0 !')

    ### Check ||∇f(x0)||
    grad0 = grad_ad(f_torch, x0)
    if np.linalg.norm(grad0) < tol: # Check optimality
        print(f'Since ||grad(x0)|| = {np.linalg.norm(grad0)} < {tol}, x0 : {x0} is optimum point !')
        list_x = [x0]; list_f = [f0]; list_grad = [grad0]
        return list_x, list_f, list_grad
    else:
        print(f'Since ||grad(x0)|| = {np.linalg.norm(grad0)} > {tol}, x0 : {x0} is not an optimum point. Optimization begins !')
        pass

    ### Initialization for searching iterations
    p_cur = None
    grad_cur = None
    
    x_new = x0
    f_new = f0
    grad_new = grad0
    
    err = 1
    k = 0

    ############## 과제용 plot을 위한 log 담기 위한 list
    list_x = [x0]
    list_f = [f0]
    list_grad = [grad0]

    ############## Searching iterations
    ###### Update info of current point
    while err > tol:
        grad_old = grad_cur
        p_old = p_cur

        x_cur = x_new
        f_cur = f_new
        grad_cur = grad_new

        ###### Line search
        ### Search direction - by Conjugate Gradient_HS method
        p_cur = search_direction_cg_hs(k, grad_old, grad_cur, p_old)
        # Search direction check
        if grad_cur@p_cur > 0: # search direction이 증가 방향이면 경고 내보내고 steepest descent direction으로 search direction 변경
            print(f'Warning : grad(x_k)·p_k={grad_cur@p_cur} > 0 : Search direction p_k would likely make function increase !')
            print(f'Warning : p_k would be replaced with steepest descent direction grad(x_k) : {-grad_cur} !')
            p_cur = search_direction_stp_descent(f_torch, x_cur)

        ### Step length - by Strong Wolfe + interpolation
        alpha_cur = wolfe_strong_interpol(f, f_torch, x_cur, f_cur, grad_cur, p_cur, 0.1)

        ##### x_new update
        k = k + 1
        x_new = x_cur + alpha_cur*p_cur
        f_new = f(x_new)
        grad_new = grad_ad(f_torch, x_new)
        err = np.linalg.norm(grad_new)
        list_x.append(x_new)
        list_f.append(f_new)
        list_grad.append(grad_new)
        print(f'x_{k} : {x_new}')
        print(f'f_{k} : {f_new}')
        print(f'norm(grad(x_{k})) : {err}')
        print(f'recent alpha : {alpha_cur}')
        print(f'recent p : {p_cur}')
        print()

    print(f'Optimization converges -> Iteration : {k} / x* : {x_new} / f(x*) : {f(x_new)} / norm(grad(x*)) : {np.linalg.norm(grad_new)} ')
    return list_x, list_f, list_grad

### Congugate Gradient Method - Fletcher Reeves(CGM-FR)
# CGM-HS보다 안정적, 적당한 속도, 더 나은 수렴안정성
def cg_fr(f, f_torch, x0, tol):
    ### Check input data type
    if (not isinstance(x0, np.ndarray)) | (x0.ndim >= 2):
        raise ValueError('Please input 1D ndarray type !!')

    ### Check f(x0)
    f0 = f(x0)
    if not np.isfinite(f0).all():
        raise ValueError('Function value at x0 is not finite. Try another x0 !')

    ### Check ||∇f(x0)||
    grad0 = grad_ad(f_torch, x0)
    if np.linalg.norm(grad0) < tol: # Check optimality
        print(f'Since ||grad(x0)|| = {np.linalg.norm(grad0)} < {tol}, x0 : {x0} is optimum point !')
        list_x = [x0]; list_f = [f0]; list_grad = [grad0]
        return list_x, list_f, list_grad
    else:
        print(f'Since ||grad(x0)|| = {np.linalg.norm(grad0)} > {tol}, x0 : {x0} is not an optimum point. Optimization begins !')
        pass

    ### Initialization for searching iterations
    p_cur = None
    grad_cur = None
    
    x_new = x0
    f_new = f0
    grad_new = grad0
    
    err = 1
    k = 0

    ############## 과제용 plot을 위한 log 담기 위한 list
    list_x = [x0]
    list_f = [f0]
    list_grad = [grad0]

    ############## Searching iterations
    ###### Update info of current point
    while err > tol:
        grad_old = grad_cur
        p_old = p_cur

        x_cur = x_new
        f_cur = f_new
        grad_cur = grad_new

        ###### Line search
        ### Search direction - by Conjugate Gradient_FR method
        p_cur = search_direction_cg_fr(k, grad_old, grad_cur, p_old)
        # Search direction check
        if grad_cur@p_cur > 0: # search direction이 증가 방향이면 경고 내보내고 steepest descent direction으로 search direction 변경
            print(f'Warning : grad(x_k)·p_k={grad_cur@p_cur} > 0 : Search direction p_k would likely make function increase !')
            print(f'Warning : p_k would be replaced with steepest descent direction grad(x_k) : {-grad_cur} !')
            p_cur = search_direction_stp_descent(f_torch, x_cur)

        ### Step length - by Strong Wolfe + interpolation
        alpha_cur = wolfe_strong_interpol(f, f_torch, x_cur, f_cur, grad_cur, p_cur, 0.1)

        ##### x_new update
        k = k + 1
        x_new = x_cur + alpha_cur*p_cur
        f_new = f(x_new)
        grad_new = grad_ad(f_torch, x_new)
        err = np.linalg.norm(grad_new)
        list_x.append(x_new)
        list_f.append(f_new)
        list_grad.append(grad_new)
        print(f'x_{k} : {x_new}')
        print(f'f_{k} : {f_new}')
        print(f'norm(grad(x_{k})) : {err}')
        print(f'recent alpha : {alpha_cur}')
        print(f'recent p : {p_cur}')
        print()

    print(f'Optimization converges -> Iteration : {k} / x* : {x_new} / f(x*) : {f(x_new)} / norm(grad(x*)) : {np.linalg.norm(grad_new)} ')
    return list_x, list_f, list_grad

### Quasi-Newton's Method - BFGS
# Newton's method와 유사한 속도, 낮은 근사 Hessian 계산량, Hessain의 PD를 보장하여 수렴안정성 비교적 높음
def quasi_newton_bfgs(f, f_torch, x0, tol):
    ### Check input data type
    if (not isinstance(x0, np.ndarray)) | (x0.ndim >= 2):
        raise ValueError('Please input 1D ndarray type !!')

    ### Check f(x0)
    f0 = f(x0)
    if not np.isfinite(f0).all():
        raise ValueError('Function value at x0 is not finite. Try another x0 !')

    ### Check ||∇f(x0)||
    grad0 = grad_ad(f_torch, x0)
    if np.linalg.norm(grad0) < tol: # Check optimality
        print(f'Since ||grad(x0)|| = {np.linalg.norm(grad0)} < {tol}, x0 : {x0} is optimum point !')
        list_x = [x0]; list_f = [f0]; list_grad = [grad0]
        return list_x, list_f, list_grad
    else:
        print(f'Since ||grad(x0)|| = {np.linalg.norm(grad0)} > {tol}, x0 : {x0} is not an optimum point. Optimization begins !')
        pass

    ### Initialization for searching iterations
    x_cur = None
    grad_cur = None
    hessian_inv_aprx_cur = np.identity(len(x0))

    x_new = x0
    f_new = f0
    grad_new = grad0
    
    tol_step_final = 1e-2*tol

    ############## 과제용 plot을 위한 log 담기 위한 list
    list_x = [x0]
    list_f = [f0]
    list_grad = [grad0]

    ############## Searching iterations
    ###### Update info of current point
    for k in range(1000):
        x_old = x_cur
        grad_old = grad_cur
        hessian_inv_aprx_old = hessian_inv_aprx_cur
        
        x_cur = x_new
        f_cur = f_new
        grad_cur = grad_new

        ###### Line search
        ### Search direction - by Quasi-Newton's(BFGS) method
        p_cur, hessian_inv_aprx_cur = search_direction_quasi_newton_bfgs(k, x_old, x_cur, grad_old, grad_cur, hessian_inv_aprx_old)

        # Search direction check
        if grad_cur@p_cur > 0: # search direction이 증가 방향이면 경고 내보내고 steepest descent direction으로 search direction 변경
            print(f'Warning : grad(x_k)·p_k={grad_cur@p_cur} > 0 : Search direction p_k would likely make function increase !')
            print(f'Warning : p_k would be replaced with steepest descent direction grad(x_k) : {-grad_cur} !')
            p_cur = search_direction_stp_descent(f_torch, x_cur)

        ### Step length - by Strong Wolfe + interpolation
        alpha_cur = wolfe_strong_interpol(f, f_torch, x_cur, f_cur, grad_cur, p_cur, 0.9)

        ##### x_new update
        x_new = x_cur + alpha_cur*p_cur
        f_new = f(x_new)
        grad_new = grad_ad(f_torch, x_new)
        list_x.append(x_new)
        list_f.append(f_new)
        list_grad.append(grad_new)

        # Convergence check for Outer loop
        r_grad_f = np.linalg.norm(grad_new)
        r_step = np.linalg.norm(x_new - x_cur)
        if k >= 2:
            if ((r_grad_f <= tol) and # ‖∇f‖도 충분히 정칙점에 도달하고
                (r_step <= tol_step_final * (1.0 + np.linalg.norm(x_new)))): # x_new도 충분히 수렴했다면
                print(f'‖∇f(x_{k+1})‖ : {r_grad_f} / tol_∇f : {tol} / ‖∆x_{k+1}‖ : {r_step} / tol_∆x : {tol_step_final}')
                break # iteration 종료하자
        print("BFGS")
        # print(f'x_{k+1} : {x_new}')
        print(f'f_{k+1} : {f_new}')
        print(f'‖∇f(x_{k+1})‖ : {r_grad_f}')
        print(f'‖∆x_{k+1}‖ : {r_step}')
        print(f'recent alpha : {alpha_cur}')
        # print(f'recent p : {p_cur}')
        print()

    print(f'BFGS \n Optimization converges -> Iteration : {k+1} / x* : {x_new} / f(x*) : {f_new} / ‖∇f(x*)‖ : {r_grad_f} ‖∆x*‖ : {r_step} \n')
    return list_x, list_f, list_grad

# ----------------------------------------------------------------------------Main Optimization Algorithm(Constrained)-------------------------------------------------------------
# ce(list) : Containing equality constraint functions
# ci(list) : Containing inequality constraint functions

### Quadratic Penalty Method(QPM)
# Heuristic method로서 penalty parameter(mu)를 키워가며 constraint를 잡는 전략
# penalty parameter(mu)가 너무 커질 경우 inner loop subproblem이 ill-conditioned하게 되어 수렴 불안정.
def qpm(f, f_torch, ce, ce_torch, ci, ci_torch, x0, inner_opt, tol):
    ### Check input data type
    if (not isinstance(ce, list)) | (not isinstance(ci, list)) | (len(ci) + len(ce) == 0):
        raise ValueError('Please input at least either one equality or inequality constraint as list type ! ; Empty list is OK as well.')

    if (not isinstance(x0, np.ndarray)) | (x0.ndim >= 2):
        raise ValueError('Please input x0 as 1D ndarray type !!')
    
    ### Check inner loop optimizer
    if inner_opt == 0:
        inner_opt = stp_descent
    elif inner_opt == 1:
        inner_opt = cg_hs
    elif inner_opt == 2:
        inner_opt = cg_fr
    elif inner_opt == 3:
        inner_opt = quasi_newton_bfgs
    else:
        raise ValueError('Please input correct integer for inner_opt ! ; 0:stp_descent, 1:cg_hs, 2:cg_fr, 3:quasi_newton_bfgs')

    ### Check f(x0)
    f0 = f(x0)
    if not np.isfinite(f0).all():
        raise ValueError('Function value at x0 is not finite. Try another x0 !')

    ### Check ci(x0) ≥ 0
    # if len(ci) >= 1: # 부등호제약조건은 feasibility 고려하고 등호제약조건은 미고려(∵ exactly하게 맞추기가 더 어려움)
    #     infeasible_ci = [ci_i(x0) for ci_i in ci if ci_i(x0) < 0] # infeasible criteria of c
    #     if len(infeasible_ci) >= 1:
    #         raise ValueError(f'Infeasible x0 for {len(infeasible_ci)} of {len(ci)} inequality constraint(s). Try feasible x0 !')

    ### Check ||∇f(x0)||
    grad0 = grad_ad(f_torch, x0)
    if np.linalg.norm(grad0) < tol: # Check optimality
        print(f'Since ||grad(x0)|| = {np.linalg.norm(grad0)} < {tol}, x0 : {x0} is optimum point !')
        # return x0

    ### constraints manipulation
    if len(ce) == 0:
        ce = [lambda x : np.float64(0.0)]
    if len(ce_torch) == 0:
        ce_torch = [lambda x : torch.tensor(0.0)]
    if len(ci) == 0:
        ci = [lambda x : np.float64(0.0)]
    if len(ci_torch) == 0:
        ci_torch = [lambda x : torch.tensor(0.0)]

    ### Initialization for searching iterations
    x_new = x0
    
    # Parameter setting for QPM
    # Qk 만들 때 필요한 quadratic penalty term of constraints 미리 정의
    sum_ce_sq = lambda x : np.sum([(ce_i(x))**2 for ce_i in ce]) # sum(c_e(x)^2) for Qk
    def sum_ce_sq_torch(x:torch.Tensor) -> torch.Tensor: # torch 버전 of sum_ce_sq
        ce_sq_torch = torch.stack([ce_torch_i(x)**2 for ce_torch_i in ce_torch])  # torch.stack 함수 쓰면 AD 그래프를 유지하면서 tensor를 쌓을 수 있음. 이거 안 쓰고 그냥 list comprehension 쓰면 AD 그래프 사라져서 AD 계산이 0이 됨.
        return ce_sq_torch.sum()
    
    sum_ci_sq = lambda x : np.sum([np.max([-ci_i(x), 0])**2 for ci_i in ci]) # sum(c_i(x)^2) for Qk
    def sum_ci_sq_torch(x:torch.Tensor) -> torch.Tensor:
        ci_sq_torch = torch.stack([torch.max(-ci_torch_i(x), torch.tensor(0))**2 for ci_torch_i in ci_torch]) # torch.stack 함수 쓰면 AD 그래프를 유지하면서 tensor를 쌓을 수 있음. 이거 안 쓰고 그냥 list comprehension 쓰면 AD 그래프 사라져서 AD 계산이 0이 됨.
        return ci_sq_torch.sum()

    f_mu = 5; mu_new = 1 # increase factor for penalty parameter mu ; mu0 = 1
    max_mu = 1e8
    f_tau = .5; tau_new = .2 # decrease factor for convergence criteria for Qk ; tau0 = .2
    tau_min = 1e-5

    ### 과제용 plot을 위한 log 담기 위한 list
    list_x = [x0]
    list_f = [f0]
    list_grad_f = [grad0]
    list_ce = [[ce_i(x0) for ce_i in ce]]
    list_ci = [[ci_i(x0) for ci_i in ci]]

    ### Outer loop begins
    for j in np.arange(100): # mu가 너무 커지면 Qk가 unstable해지기 때문에 어차피 finite한 iterations 내에서 쇼부를 봐야 한다.
        # Finding a x*_k in Qk
        x_cur = x_new # x_k
        mu_cur = mu_new; print(f'mu_{j} = {mu_cur}') # penalty parameter update(increase)
        tau_cur = tau_new; print(f'tau_{j} = {tau_cur}') # Qk convergence criteria update(decrease)
        Q_cur = lambda x : f(x) + .5*mu_cur*sum_ce_sq(x) + .5*mu_cur*sum_ci_sq(x) # quadratic penalty function Qk at x_k
        Q_cur_torch = lambda x : f_torch(x) + .5*mu_cur*sum_ce_sq_torch(x) + .5*mu_cur*sum_ci_sq_torch(x) # 
        log_inner = inner_opt(Q_cur, Q_cur_torch, x_cur, tau_cur) # Solving ∇Qk(x*_k) ≤ tau_k ; Inner loop
        x_new = log_inner[0][-1]; f_new = f(x_new); grad_f_new = grad_ad(f_torch, x_new); ce_new = [ce_i(x_new) for ce_i in ce]; ci_new = [ci_i(x_new) for ci_i in ci]
        mu_new = min(mu_cur*f_mu, max_mu)
        tau_new = max(tau_cur*f_tau, tau_min)

        # Convergence check of x* ... QPM을 위한 convergence 기준은 명확한 게 없음. 애초에 휴리스틱 알고리즘이라 근본이 없는 놈이라 그럼.
        move_x = np.linalg.norm(x_new - x_cur) # convergence criteria of x*_k
        violated_ce = [ce_new_i for ce_new_i in ce_new if np.abs(ce_new_i) > 1e-4] # violation criteria of ce(x*_k)
        violated_ci = [ci_new_i for ci_new_i in ci_new if ci_new_i < -1e-4] # violation criteria of ci(x*_k)
        if (move_x < 1e-4) & (len(violated_ci) == 0) & (len(violated_ce) == 0):
            done = True # flag for termination of outerloop
        else:
            done = False

        print(f'{j+1}-th outer loop : Inner loop converges at {len(log_inner[0]) - 1} iteration(s) ...')
        print(f'|x_{j+1} - x_{j}| = {move_x}')
        print(f'# of violated ce constraints : {len(violated_ce)}, violation : {np.sum(violated_ce)}')
        print(f'# of violated ci constraints : {len(violated_ci)}, violation : {np.sum(violated_ci)}')
        print(f'\n------------------------------------------------------------- Outer loop ----------------------------------------------------------------\n')

        list_x.append(x_new)
        list_f.append(f_new)
        list_grad_f.append(grad_f_new)
        list_ce.append(ce_new)
        list_ci.append(ci_new)

        if done:
            print(f'Outer loop converged at {j+1} iteration(s) !')
            print(f'iter = {j+1} x* = {x_new}, f(x*) = {f(x_new)}, |ce(x*)|₁ = {np.sum(np.abs(ce_new))}, |ci(x*)|₁ = {np.sum(np.abs(ci_new))}')
            return list_x, list_f, list_grad_f, list_ce, list_ci
    else:
        return list_x, list_f, list_grad_f, list_ce, list_ci

### Augmented Lagrangian Method(ALM)
# KKT conditions based on Lagrangian function 개념을 활용하여 비교적 작은 penalty method(mu, rho)로도 안정적 수렴 가능.
def alm(f, f_torch, ce, ce_torch, ci, ci_torch, x0, inner_opt, tol):
    ### Check input data type
    if (not isinstance(ce, list)) | (not isinstance(ci, list)) | (len(ci) + len(ce) == 0):
        raise ValueError('Please input at least either one equality or inequality constraint as list type ! ; Empty list is OK as well.')

    if (not isinstance(x0, np.ndarray)) | (x0.ndim >= 2):
        raise ValueError('Please input x0 as 1D ndarray type !!')

    ### Check inner loop optimizer
    if inner_opt == 0:
        inner_opt = stp_descent
    elif inner_opt == 1:
        inner_opt = cg_hs
    elif inner_opt == 2:
        inner_opt = cg_fr
    elif inner_opt == 3:
        inner_opt = quasi_newton_bfgs
    else:
        raise ValueError('Please input correct integer for inner_opt ! ; 0:stp_descent, 1:cg_hs, 2:cg_fr, 3:quasi_newton_bfgs')

    ### Check f(x0)
    f0 = f(x0)
    if not np.isfinite(f0).all():
        raise ValueError('Function value at x0 is not finite. Try another x0 !')

    ### Check ci(x0) ≥ 0
    if len(ci) >= 1: # 부등호제약조건은 feasibility 고려하고 등호제약조건은 미고려(∵ exactly하게 맞추기가 더 어려움)
        infeasible_ci = [ci_j(x0) for ci_j in ci if ci_j(x0) < 0] # infeasible criteria of c
        if len(infeasible_ci) >= 1:
            raise ValueError(f'Infeasible x0 for {len(infeasible_ci)} of {len(ci)} inequality constraint(s). Try feasible x0 !')

    ### Check ||∇f(x0)||
    grad0 = grad_ad(f_torch, x0)
    if np.linalg.norm(grad0) < tol: # Check optimality
        print(f'Since ||grad(x0)|| = {np.linalg.norm(grad0)} < {tol}, x0 : {x0} is optimum point !')
        # return x0

    ### constraints manipulation
    if len(ce) == 0:
        ce = [lambda x : np.float64(0.0)]
    if len(ce_torch) == 0:
        ce_torch = [lambda x : x.sum()*0.0] # x.sum() 안 쓰고 그냥 0.0만 써서 상수함수 만들면 torch의 AD 사용이 불가함. 따라서 실제로는 상수 함수더라도 x에 대한 함수처럼 보이게끔 표기해야함(꼼수).
    if len(ci) == 0:
        ci = [lambda x : np.float64(0.0)]
    if len(ci_torch) == 0:
        ci_torch = [lambda x : x.sum()*0.0] # x.sum() 안 쓰고 그냥 0.0만 써서 상수함수 만들면 torch의 AD 사용이 불가함. 따라서 실제로는 상수 함수더라도 x에 대한 함수처럼 보이게끔 표기해야함(꼼수).

    ### Initialization for searching iterations
    x_new = x0

    ### Parameter setting for ALM
    # Final tolerance for outer loop
    tol_eq_final = 1e-5 # equality feasibility
    tol_ineq_final = 1e-5 # inequality feasibility
    tol_opt_final = tol # optimality (∥∇L_A∥∞)
    tol_step_final = 1e-3*tol # step size (relative factor 포함 권장)

    # Initial tolearnce of constraints for outer loop(점차 tighten)
    tol_eq   = 1e-3
    tol_ineq = 1e-3

    # Tolerance update scheme for inner loop : tau_k = max(omega_min, omega0 * (omega_decay)^k)
    omega0 = 1e-2
    omega_decay = 0.5
    omega_min = tol_opt_final
    tau = omega0   # == tau_0

    # Penalty parameter increase factor / upper bound
    factor_mu  = 5.0
    factor_rho = 5.0
    mu_max   = 1e8
    rho_max  = 1e8

    lmbda = np.array([0.0]*len(ce)) # initial lagrange multipliers of equality constraints for Lk
    nu = np.array([0.0]*len(ci)) # initial lagrange multipliers of inequality constraints
    mu = 1 # increase factor for equality constraint penalty parameter mu ; mu0 = 1
    rho = 1 # increase factor for inequality constraint penalty parameter rho ; rho0 = 1

    ### 과제용 plot을 위한 log 담기 위한 list
    list_x = [x0]
    list_f = [f0]
    list_grad = [grad0]
    list_ce = [[ce_i(x0) for ce_i in ce]]
    list_ci = [[ci_i(x0) for ci_i in ci]]
    list_lmbda = [lmbda]
    list_nu = [nu]

    ### Outer loop begins
    for k in np.arange(100):
        x_cur = x_new # x_k
        print(f'mu_{k} = {mu}')
        print(f'rho_{k} = {rho}')
        print(f'tau_{k} = {tau}')

        # penalty term update
        penalty_ce = lambda x : -lmbda@np.array([ce_j(x) for ce_j in ce]) + .5*mu*np.sum(np.array([ce_j(x)**2 for ce_j in ce]))
        def penalty_ce_torch(x:torch.Tensor) -> torch.Tensor:
            lmbda_torch = torch.tensor(lmbda)
            vals = torch.stack([ce_torch_j(x) for ce_torch_j in ce_torch]) # torch.stack 함수 써서 AD graph 안 끊기게 함
            return -lmbda_torch@vals + .5*mu*(vals**2).sum()
        
        penalty_ci = lambda x : (-nu@np.array([ci_j(x) - np.maximum(ci_j(x) - nu[j]/rho, 0) for j, ci_j in enumerate(ci)]) +
                                    .5*rho*np.sum(np.array([(ci_j(x) - np.maximum(ci_j(x) - nu[j]/rho, 0))**2 for j, ci_j in enumerate(ci)])))
        def penalty_ci_torch(x:torch.Tensor) -> torch.Tensor:
            nu_torch = torch.tensor(nu)
            rho_torch = torch.tensor(rho)
            vals = torch.stack([ci_torch_j(x) for ci_torch_j in ci_torch]) # torch.stack 함수 써서 AD graph 안 끊기게 함
            zeros = torch.zeros_like(vals)
            result = -nu_torch@(vals - torch.maximum(vals - nu_torch/rho_torch, zeros)) + .5*rho_torch*((vals - torch.maximum(vals - nu_torch/rho_torch, zeros))**2).sum()
            return result

        LA_cur = lambda x : f(x) + penalty_ce(x) + penalty_ci(x) # augmented lagrangian function LAk at x_k
        LA_cur_torch = lambda x : f_torch(x) + penalty_ce_torch(x) + penalty_ci_torch(x)
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            log_inner = inner_opt(LA_cur, LA_cur_torch, x_cur, tau) # solving ∇LAk(x*_k) ≤ tau_k ; Inner loop
        x_new = log_inner[0][-1]
        f_new = f(x_new)
        ce_new = np.array([ce_j(x_new) for ce_j in ce])
        ci_new = np.array([ci_j(x_new) for ci_j in ci])
        # --- ∇L update ---
        # - multiplier updates (AFTER computing ce_new, ci_new) -
        lmbda = lmbda - mu * ce_new
        nu = np.maximum(nu - rho * ci_new, 0.0)
        grad_f_new = grad_ad(f_torch, x_new)
        sum_lmbdaxgrad_ce_new = np.array([lmbda_j*grad_ad(ce_torch_j, x_new) for lmbda_j, ce_torch_j in zip(lmbda, ce_torch)]).sum(axis=0)
        sum_nuxgrad_ci_new = np.array([nu_j*grad_ad(ci_torch_j, x_new) for nu_j, ci_torch_j in zip(nu, ci_torch)]).sum(axis=0)
        # x_new를 grad_f(원래 문제의 목적함수의 gradient)와 grad_ce(원래 문제의 등호제약함수), grad_ci(원래 문제의 부등호제약함수)에 넣어서 원래 문제의 ∇L(x_new)를 grad_L로 구해야 함
        # grad_LA_new는 augmented Lagrangian function의 gradient라 penalty parameter(mu, rho)가 다 반영되어있는 기울기임. 
        # 내부 solver(quasi newton bfgs) 입장에서나 의미 있는 기울기지 원래 문제에 있어서 아무 의미 없는 기울기.
        # 따라서 convergence check할 때는 원래 문제에 대한 grad_L을 기준으로 해야 함.
        grad_L_new = grad_f_new - sum_lmbdaxgrad_ce_new - sum_nuxgrad_ci_new

        # residual(잔차) 계산
        r_ce = np.max(np.abs(ce_new)) # 등호제약조건 잔차(위반)
        r_ci = np.max(np.maximum(-ci_new, 0)) # 부등호제약조건 잔차(위반)
        r_grad_L = np.linalg.norm(grad_L_new, ord=np.inf) # ∇L 수준
        r_step = np.linalg.norm(x_new - x_cur) # x_new - x_cur 거리

        # Convergence check for Outer loop
        if ((r_ce <= tol_eq_final) & # 등호제약조건 위반이 충분히 작고
            (r_ci <= tol_ineq_final) & # 부등호제약조건 위반도 충분히 작고
            (r_grad_L <= tol_opt_final) & # ∇L도 충분히 정칙점에 도달했고
            # (r_step <= tol_step_final * (1.0 + np.linalg.norm(x_new)))): # x_new도 충분히 수렴했다면
            (r_step <= tol_step_final)): # x_new도 충분히 수렴했다면
            done = True # outer loop 종료 flag 마킹하자
        else:
            done = False

        # ---- penalty parameter update(keep or increase) ----
        # ce
        if r_ce <= tol_eq:
            mu = mu # 등호제약조건 위반이 충분히 작다면 페널티 파라미터를 그대로 두자
        else:
            mu = min(factor_mu*mu, mu_max) # 등호제약조건 위반이 크다면 페널티 파라미터를 증가시키자

        # ci
        if r_ci <= tol_ineq:
            rho = rho # 부등호제약조건 위반이 충분히 작다면 페널티 파라미터를 그대로 두자
        else:
            rho = min(factor_rho*rho, rho_max) # 부등호제약조건 위반이 크다면 페널티 파라미터를 증가시키자

        # ---- tolerance update ----
        # If 제약조건 잔차 ≈ 0 and ∇L ≈ 0 -> 제약조건 tolerance를 조금 더 빡세게 두자(감소시키자)
        if (r_ce <= 0.3*tol_eq) & (r_ci <= 0.3*tol_ineq) & (r_grad_L <= 0.3*tau):
            tol_eq = max(tol_eq_final,   0.5*tol_eq)
            tol_ineq = max(tol_ineq_final, 0.5*tol_ineq)

        # 내부 허용치 스케줄(항상 단조 감소)
        tau = max(omega_min, omega_decay*tau)

        # ---------------------------------------- print log ----------------------------------------
        print(f'{k+1}-th outer loop : Inner loop converges at {len(log_inner[0]) - 1} iteration(s) ...')
        print(f'|x_{k+1} - x_{k}| = {r_step}')
        print(f'f(x_{k+1}) = {f_new}')
        print(f'|∇L(x_{k+1})| = {r_grad_L}')
        print(f'Max violation of equality constraints : {r_ce}')
        print(f'Max violation of inequality constraints : {r_ci}')
        print(f'\n------------------------------------------------------------- Outer loop ----------------------------------------------------------------\n')

        list_x.append(x_new)
        list_f.append(f_new)
        list_grad.append(grad_L_new)
        list_ce.append(ce_new)
        list_ci.append(ci_new)
        list_lmbda.append(lmbda)
        list_nu.append(nu)

        if done:
            print(f'Outer loop converges at {k+1} iteration(s) !')
            print(f'iter = {k+1} x* = {x_new}, f(x*) = {f_new}, |∇L(x*)| = {r_grad_L}, max(ce(x*)) = {r_ce}, max(ci(x*)) = {r_ci}')
            return list_x, list_f, list_grad, list_ce, list_ci, list_lmbda, list_nu
        
    print(f'Outer loop terminates at {k+1}(max) iteration(s) !')
    print(f'iter = {k+1} x* = {x_new}, f(x*) = {f_new}, |∇L(x*)| = {r_grad_L}, max(ce(x*)) = {r_ce}, max(ci(x*)) = {r_ci}')
    return list_x, list_f, list_grad, list_ce, list_ci, list_lmbda, list_nu

### Augmented Lagrangian Method(ALM) for SQP(Sequential Quadratic Programming)
# SQP 알고리즘의 내부(중간) 알고리즘으로 사용하고자 따로 만듦. 구조는 위의 ALM이랑 동일
def alm4sqp(f, f_torch, ce, ce_torch, ci, ci_torch, x0, lmbda0, nu0, inner_opt, tol): # 그냥 alm과는 조금 다르게 sqp의 outer iteration으로부터 lmbda0와 nu0를 받음
    '''
    alm4sqp solves QP_k for p_k* which makes x_(k+1) = x_k + α_k*p_k*
    f : Quadratic objective function of QP_k
    ce : Linear equlaity constraint function of QP_k
    ci : Linear inequlaity constraint function of QP_k
    x0 : Initial guess of search direction(p_k0) that QP_k tries to find out. It is expected to be received as p_(k-1)* from SQP loop ... hot start strategy
    lmbda0 : Initial guess of Lagrange multipliers for ce. It is expected to be received as lmbda_(k-1)* from SQP loop
    nu0 : Initial guess of Lagrange multipliers for ci. It is expected to be received as nu_(k-1)* from SQP loop
    '''
    ### Check input data type
    if (not isinstance(ce, list)) | (not isinstance(ci, list)) | (len(ci) + len(ce) == 0):
        raise ValueError('Please input at least either one equality or inequality constraint as list type ! ; Empty list is OK as well.')

    if (not isinstance(x0, np.ndarray)) | (x0.ndim >= 2):
        raise ValueError('Please input x0 as 1D ndarray type !!')

    ### Check inner loop optimizer
    if inner_opt == 0:
        inner_opt = stp_descent
    elif inner_opt == 1:
        inner_opt = cg_hs
    elif inner_opt == 2:
        inner_opt = cg_fr
    elif inner_opt == 3:
        inner_opt = quasi_newton_bfgs
    else:
        raise ValueError('Please input correct integer for inner_opt ! ; 0:stp_descent, 1:cg_hs, 2:cg_fr, 3:quasi_newton_bfgs')

    ### Check f(x0)
    f0 = f(x0)
    if not np.isfinite(f0).all():
        raise ValueError('Function value at x0 is not finite. Try another x0 !')

    # ### Check ci(x0) ≥ 0 # SQP를 위한 ALM에서는 infeasible initial point 허용하기로 하자.
    # if len(ci) >= 1: # 부등호제약조건은 feasibility 고려하고 등호제약조건은 미고려(∵ exactly하게 맞추기가 더 어려움)
    #     infeasible_ci = [ci_j(x0) for ci_j in ci if ci_j(x0) < 0] # infeasible criteria of c
    #     if len(infeasible_ci) >= 1:
    #         raise ValueError(f'Infeasible x0 for {len(infeasible_ci)} of {len(ci)} inequality constraint(s). Try feasible x0 !')

    ### Check ||∇f(x0)||
    grad0 = grad_ad(f_torch, x0)
    if np.linalg.norm(grad0) < tol: # Check optimality
        # print(f'Since ||grad(x0)|| = {np.linalg.norm(grad0)} < {tol}, x0 : {x0} is optimum point !')
        # return x0
        pass
    
    ### constraints manipulation
    if len(ce) == 0:
        ce = [lambda x : np.float64(0.0)]
    if len(ce_torch) == 0:
        ce_torch = [lambda x : x.sum()*0.0] # x.sum() 안 쓰고 그냥 0.0만 써서 상수함수 만들면 torch의 AD 사용이 불가함. 따라서 실제로는 상수 함수더라도 x에 대한 함수처럼 보이게끔 표기해야함(꼼수).
    if len(ci) == 0:
        ci = [lambda x : np.float64(0.0)]
    if len(ci_torch) == 0:
        ci_torch = [lambda x : x.sum()*0.0] # x.sum() 안 쓰고 그냥 0.0만 써서 상수함수 만들면 torch의 AD 사용이 불가함. 따라서 실제로는 상수 함수더라도 x에 대한 함수처럼 보이게끔 표기해야함(꼼수).

    ### Initialization for searching iterations
    x_new = x0

    ### Parameter setting for ALM
    # Final tolerance for outer loop
    tol_eq_final = 1e-5 # equality feasibility
    tol_ineq_final = 1e-5 # inequality feasibility
    tol_opt_final = tol # optimality (∥∇L_A∥∞)
    tol_step_final = 1e-3*tol # step size (relative factor 포함 권장)

    # Initial tolearnce of constraints for outer loop(점차 tighten)
    tol_eq   = 1e-3
    tol_ineq = 1e-3

    # Tolerance update scheme for inner loop : tau_k = max(omega_min, omega0 * (omega_decay)^k)
    omega0 = 1e-2
    omega_decay = 0.5
    omega_min = tol_opt_final
    tau = omega0   # == tau_0

    # Penalty parameter increase factor / upper bound
    factor_mu  = 5.0
    factor_rho = 5.0
    mu_max   = 1e8
    rho_max  = 1e8

    lmbda = lmbda0 # initial lagrange multipliers of equality constraints for Lk
    nu = nu0 # initial lagrange multipliers of inequality constraints
    mu = 1 # increase factor for equality constraint penalty parameter mu ; mu0 = 1
    rho = 1 # increase factor for inequality constraint penalty parameter rho ; rho0 = 1

    ### 과제용 plot을 위한 log 담기 위한 list
    list_x = [x0]
    list_f = [f0]
    list_grad = [grad0]
    list_ce = [[ce_i(x0) for ce_i in ce]]
    list_ci = [[ci_i(x0) for ci_i in ci]]
    list_lmbda = [lmbda]
    list_nu = [nu]

    ### Outer loop begins
    for k in np.arange(100):
        x_cur = x_new # x_k
        # print(f'mu_{k} = {mu}')
        # print(f'rho_{k} = {rho}')
        # print(f'tau_{k} = {tau}')
        
        # penalty term update
        penalty_ce = lambda x : -lmbda@np.array([ce_j(x) for ce_j in ce]) + .5*mu*np.sum(np.array([ce_j(x)**2 for ce_j in ce]))
        def penalty_ce_torch(x:torch.Tensor) -> torch.Tensor:
            lmbda_torch = torch.tensor(lmbda)
            vals = torch.stack([ce_torch_j(x) for ce_torch_j in ce_torch]) # torch.stack 함수 써서 AD graph 안 끊기게 함
            return -lmbda_torch@vals + .5*mu*(vals**2).sum()
        
        penalty_ci = lambda x : (-nu@np.array([ci_j(x) - np.maximum(ci_j(x) - nu[j]/rho, 0) for j, ci_j in enumerate(ci)]) +
                                    .5*rho*np.sum(np.array([(ci_j(x) - np.maximum(ci_j(x) - nu[j]/rho, 0))**2 for j, ci_j in enumerate(ci)])))
        def penalty_ci_torch(x:torch.Tensor) -> torch.Tensor:
            nu_torch = torch.tensor(nu)
            rho_torch = torch.tensor(rho)
            vals = torch.stack([ci_torch_j(x) for ci_torch_j in ci_torch]) # torch.stack 함수 써서 AD graph 안 끊기게 함
            zeros = torch.zeros_like(vals)
            result = -nu_torch@(vals - torch.maximum(vals - nu_torch/rho_torch, zeros)) + .5*rho_torch*((vals - torch.maximum(vals - nu_torch/rho_torch, zeros))**2).sum()
            return result

        LA_cur = lambda x : f(x) + penalty_ce(x) + penalty_ci(x) # augmented lagrangian function LAk at x_k
        LA_cur_torch = lambda x : f_torch(x) + penalty_ce_torch(x) + penalty_ci_torch(x)
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            log_inner = inner_opt(LA_cur, LA_cur_torch, x_cur, tau) # solving ∇LAk(x*_k) ≤ tau_k ; Inner loop
        # log_inner = inner_opt(LA_cur, LA_cur_torch, x_cur, tau) # 디버깅용
        x_new = log_inner[0][-1]
        f_new = f(x_new)
        ce_new = np.array([ce_j(x_new) for ce_j in ce])
        ci_new = np.array([ci_j(x_new) for ci_j in ci])
        # --- ∇L update ---
        # - multiplier updates (AFTER computing ce_new, ci_new) -
        lmbda = lmbda - mu * ce_new
        nu = np.maximum(nu - rho * ci_new, 0.0)
        grad_f_new = grad_ad(f_torch, x_new)
        sum_lmbdaxgrad_ce_new = np.array([lmbda_j*grad_ad(ce_torch_j, x_new) for lmbda_j, ce_torch_j in zip(lmbda, ce_torch)]).sum(axis=0)
        sum_nuxgrad_ci_new = np.array([nu_j*grad_ad(ci_torch_j, x_new) for nu_j, ci_torch_j in zip(nu, ci_torch)]).sum(axis=0)
        # x_new를 grad_f(원래 문제의 목적함수의 gradient)와 grad_ce(원래 문제의 등호제약함수), grad_ci(원래 문제의 부등호제약함수)에 넣어서 원래 문제의 ∇L(x_new)를 grad_L로 구해야 함
        # grad_LA_new는 augmented Lagrangian function의 gradient라 penalty parameter(mu, rho)가 다 반영되어있는 기울기임. 
        # 내부 solver(quasi newton bfgs) 입장에서나 의미 있는 기울기지 원래 문제에 있어서 아무 의미 없는 기울기.
        # 따라서 convergence check할 때는 원래 문제에 대한 grad_L을 기준으로 해야 함.
        grad_L_new = grad_f_new - sum_lmbdaxgrad_ce_new - sum_nuxgrad_ci_new

        # residual(잔차) 계산
        r_ce = np.max(np.abs(ce_new)) if len(ce_new) >= 1 else 0 # 등호제약조건 잔차(위반)
        r_ci = np.max(np.maximum(-ci_new, 0)) if len(ci_new) >= 1 else 0 # 부등호제약조건 잔차(위반)
        r_grad_L = np.linalg.norm(grad_L_new, ord=np.inf) # ∇L 수준
        r_step = np.linalg.norm(x_new - x_cur) # x_new - x_cur 거리

        # Convergence check for Outer loop
        if ((r_ce <= tol_eq_final) & # 등호제약조건 위반이 충분히 작고
            (r_ci <= tol_ineq_final) & # 부등호제약조건 위반도 충분히 작고
            (r_grad_L <= tol_opt_final) & # ∇L도 충분히 정칙점에 도달했고
            # (r_step <= tol_step_final * (1.0 + np.linalg.norm(x_new)))): # x_new도 충분히 수렴했다면
            (r_step <= tol_step_final)): # x_new도 충분히 수렴했다면
            done = True # outer loop 종료 flag 마킹하자
        else:
            done = False

        # ---- penalty parameter update(keep or increase) ----
        # ce
        if r_ce <= tol_eq:
            mu = mu # 등호제약조건 위반이 충분히 작다면 페널티 파라미터를 그대로 두자
        else:
            mu = min(factor_mu*mu, mu_max) # 등호제약조건 위반이 크다면 페널티 파라미터를 증가시키자

        # ci
        if r_ci <= tol_ineq:
            rho = rho # 부등호제약조건 위반이 충분히 작다면 페널티 파라미터를 그대로 두자
        else:
            rho = min(factor_rho*rho, rho_max) # 부등호제약조건 위반이 크다면 페널티 파라미터를 증가시키자

        # ---- tolerance update ----
        # If 제약조건 잔차 ≈ 0 and ∇L ≈ 0 -> 제약조건 tolerance를 조금 더 빡세게 두자(감소시키자)
        if (r_ce <= 0.3*tol_eq) & (r_ci <= 0.3*tol_ineq) & (r_grad_L <= 0.3*tau):
            tol_eq = max(tol_eq_final,   0.5*tol_eq)
            tol_ineq = max(tol_ineq_final, 0.5*tol_ineq)

        # 내부 허용치 스케줄(항상 단조 감소)
        tau = max(omega_min, omega_decay*tau)

        # ---------------------------------------- print log ----------------------------------------
        x_str = ", ".join([f"{xi:.8f}" for xi in x_new])
        print("\n log - ALM")
        print(f"‖∆p‖ = {r_step:.2e}, "
            f"p{k+1:02d} = [{x_str}] | \n"
            f"Q_QPk = {f_new:.4e}, "
            f"‖∇L‖ = {r_grad_L:.2e}, "
            f"‖ce_QPk‖∞ = {r_ce:.2e}, "
            f"‖ci_QPk‖∞ = {r_ci:.2e}, "
            f"‖λ_ce_QPk‖ = {np.linalg.norm(lmbda):.2e}, "
            f"‖ν_ci_QPk‖ = {np.linalg.norm(nu):.2e}\n")
        list_x.append(x_new)
        list_f.append(f_new)
        list_grad.append(grad_L_new)
        list_ce.append(ce_new)
        list_ci.append(ci_new)
        list_lmbda.append(lmbda)
        list_nu.append(nu)

        if done:
            return list_x, list_f, list_grad, list_ce, list_ci, list_lmbda, list_nu

    return list_x, list_f, list_grad, list_ce, list_ci, list_lmbda, list_nu

### SQP(Sequential Quadratic Programming) using ALM as a QP_k solver
# unconstrained_solver ⊂ ALM ⊂ SQP ... 3중 loop 구조여서 엄청 느림. 설계변수/제약함수 개수 많은 문제 풀 때 work poorly
def sqp(f, f_torch, ce, ce_torch, ci, ci_torch, x0, maxiter=100, inner_opt=3, tol=1e-6, tol_inter=1e-4):
    '''
    Sequential Quadratic Programming(SQP) using ALM as an intermediate algorithm for solving QP subproblem.  

    Args:
        f (callable) : objective function(output \: single scalar)
        ce (list[callable]) : Equality constraint functions
        ci (list[callable]) : Inequality constraint functions
        x0 (1D ndarray) : Initial guess(# of it should be same wi
        th input variables of f)
        inner_opt (int, optional) : Inner optimizer for ALM(defau
        lt \: 3)  
            (0 \: SDM 1 \: CGM_HS 2 \: CGM_FR 3 \: BFGS)

        tol (float, optional) : Convergence tolerance ε for outer
         iteration(SQP).  
            ∇L_k < ε (k \: iteration of SQP)

        tol_inter (float, optional) : Convergence tolerance ε for
         intermediate iteration(ALM).  
            ∇L_A_j < ε (j \: iteration of ALM)


    Returns:
        log (list[list]) : Optimization log containing:
         - list_x (list) : Solution log
         - list_f (list) : Final objective value log
         - list_grad_f (list) : Gradient of obj function log
         - list_grad_L (list) : Gradient of Lagrangian log
         - list_ce (list) : Maximum violation of equality constraint value log
         - list_ci (list) : Maximum violation of inequality constraint value log
         - list_lmbda (list) : Lagrange multipliers for equality constraints log
         - list_nu (list) : Lagrange multipliers for inequality constraints log
    '''
    ### Check input data type
    if (not isinstance(ce, list)) | (not isinstance(ci, list)) | (len(ci) + len(ce) == 0):
        raise ValueError('Please input at least either one equality or inequality constraint as list type ! ; Empty list is OK as well.')

    if (not isinstance(x0, np.ndarray)) | (x0.ndim >= 2):
        raise ValueError('Please input x0 as 1D ndarray type !!')

    ### Check f(x0)
    f0 = f(x0)
    if not np.isfinite(f0).all():
        raise ValueError('Function value at x0 is not finite. Try another x0 !')

    ### Check ||∇f(x0)||
    grad0 = grad_ad(f_torch, x0)
    if np.linalg.norm(grad0) < tol: # Check optimality
        print(f'Since ||grad(x0)|| = {np.linalg.norm(grad0)} < {tol}, x0 : {x0} is optimum point !')
        # return x0

    ### constraints manipulation
    if len(ce) == 0:
        ce = [lambda x : np.float64(0.0)]
    if len(ce_torch) == 0:
        ce_torch = [lambda x : x.sum()*0.0] # x.sum() 안 쓰고 그냥 0.0만 써서 상수함수 만들면 torch의 AD 사용이 불가함. 따라서 실제로는 상수 함수더라도 x에 대한 함수처럼 보이게끔 표기해야함(꼼수).
    if len(ci) == 0:
        ci = [lambda x : np.float64(0.0)]
    if len(ci_torch) == 0:
        ci_torch = [lambda x : x.sum()*0.0] # x.sum() 안 쓰고 그냥 0.0만 써서 상수함수 만들면 torch의 AD 사용이 불가함. 따라서 실제로는 상수 함수더라도 x에 대한 함수처럼 보이게끔 표기해야함(꼼수).

    ### Initialization for searching iterations
    x_new = x0
    f_new = f0
    ce_new = [ce_j(x_new) for ce_j in ce]
    ci_new = [ci_j(x_new) for ci_j in ci]
    grad_f_new = grad0
    grad_ce_new = [grad_ad(ce_torch_j, x0) for ce_torch_j in ce_torch]
    grad_ci_new = [grad_ad(ci_torch_j, x0) for ci_torch_j in ci_torch]
    lmbda_new = np.array([0.0]*len(ce)) # initial lagrange multipliers of equality constraints for Lk
    nu_new = np.array([0.0]*len(ci)) # initial lagrange multipliers of inequality constraints for Lk

    # tolerances
    tol_opt_final   = tol
    tol_eq_final    = 1e-6
    tol_ineq_final  = 1e-6
    tol_step_final  = 1e-3*tol

    ### 과제용 plot을 위한 log 담기 위한 list
    list_x = [x0]
    list_f = [f0]
    list_grad_f = [grad0] # 굳이 저장 안 해도 됨(생략 가능)
    list_grad_L = [grad0]
    list_ce = [[ce_i(x0) for ce_i in ce]]
    list_ci = [[ci_i(x0) for ci_i in ci]]
    list_lmbda = [lmbda_new]
    list_nu = [nu_new]

    ### Outer loop begins
    for k in np.arange(maxiter): # max iteration of outer loop = 100
        
        # ---------------- Update QPk ----------------
        x_k = x_new # x_k
        ce_k = ce_new
        ci_k = ci_new
        grad_f_k = grad_f_new
        grad_ce_k = grad_ce_new
        grad_ci_k = grad_ci_new
        lmbda_k = lmbda_new
        nu_k = nu_new
        # Damped-BFGS Bk update
        if k == 0:
            # 첫 반복: H는 그냥 I로 시작 (또는 스케일 조정된 I)
            B_k = np.eye(len(x_new))
        else:
            sy = s_k @ y_k # 스칼라
            sBs = s_k @ (B_k @ s_k) # 스칼라
            theta = 0.8*sBs/(sBs - sy) if (sy < 0.2*sBs) else 1 # 스칼라
            r_k = theta*y_k + (1 - theta)*(B_k @ s_k) # (n, ) ndarray
            BssB = (B_k @ s_k).reshape(-1, 1) @ (s_k @ B_k).reshape(1, -1) # (n, n) ndarray
            rr = r_k.reshape(-1, 1) @ r_k.reshape(1, -1) # (n, n) ndarray
            sr = s_k @ r_k # 스칼라
            ### 0으로 나눠지는 경우 BFGS update skip하고 이전 B_k 유지
            if ((np.abs(sBs) > 1e-3) & (np.abs(sr) > 1e-3)):
                B_k = B_k - BssB/sBs + rr/sr
            else:
                B_k = B_k
        # Objective function of QPk subproblem
        Q_QPk = lambda p : .5*p@(B_k@p) + grad_f_k@p
        def Q_QPk_torch(p:torch.Tensor) -> torch.Tensor: # torch 함수 추가 1 --------------------------------------------------
            B_k_torch = torch.tensor(B_k)
            grad_f_k_torch = torch.tensor(grad_f_k)
            return .5*p@(B_k_torch@p) + grad_f_k_torch@p # torch 함수 추가 1 끝 --------------------------------------------------
        # Equality constraint function of QPk subproblem
        ce_QPk = [lambda p, j=j, a=grad_ce_k_j: a@p + ce_k[j] for j, grad_ce_k_j in enumerate(grad_ce_k)] # 클로저 late binding 문제 조심
        ce_QPk_torch = [] # torch 함수 추가 2 --------------------------------------------------
        for ce_k_i, grad_ce_k_i in zip(ce_k, grad_ce_k):
            def ce_QPk_torch_i(p:torch.Tensor, a=grad_ce_k_i, ce_k_i=ce_k_i) -> torch.Tensor: # 클로저 late binding 문제 조심
                a_torch_i = torch.tensor(a)
                ce_k_torch_i = torch.tensor(ce_k_i)
                return a_torch_i@p + ce_k_torch_i
            ce_QPk_torch.append(ce_QPk_torch_i) # torch 함수 추가 2 끝 --------------------------------------------------
        # Inequality constraint function of QPk subproblem
        ci_QPk = [lambda p, j=j, a=grad_ci_k_j: a@p + ci_k[j] for j, grad_ci_k_j in enumerate(grad_ci_k)] # 클로저 late binding 문제 조심
        ci_QPk_torch = [] # torch 함수 추가 3 --------------------------------------------------
        for ci_k_i, grad_ci_k_i in zip(ci_k, grad_ci_k):
            def ci_QPk_torch_i(p:torch.Tensor, a=grad_ci_k_i, ci_k_i=ci_k_i) -> torch.Tensor: # 클로저 late binding 문제 조심
                a_torch_i = torch.tensor(a)
                ci_k_torch_i = torch.tensor(ci_k_i)
                return a_torch_i@p + ci_k_torch_i
            ci_QPk_torch.append(ci_QPk_torch_i) # torch 함수 추가 3 끝 --------------------------------------------------

        # Hot initial start(p0_QPk) for accelerating ALM
        if k == 0:
            p0_QPk = np.array([0.0]*len(x_k))
        else:
            p0_QPk = p_new

        # ALM for solving QPk subproblem
        # with io.StringIO() as buf, contextlib.redirect_stdout(buf): # Intermediate loop(ALM) ... 실전수행용
            # log_inter = alm4sqp(Q_QPk, Q_QPk_torch, ce_QPk, ce_QPk_torch, ci_QPk, ci_QPk_torch, p0_QPk, lmbda_k, nu_k, inner_opt, tol_inter)
        log_inter = alm4sqp(Q_QPk, Q_QPk_torch, ce_QPk, ce_QPk_torch, ci_QPk, ci_QPk_torch, p0_QPk, lmbda_k, nu_k, inner_opt, tol_inter) # Intermediate loop(ALM) ... 디버깅용
        
        # ---------------- Update x info ----------------
        p_new = log_inter[0][-1] # search direction p_k
        lmbda_new = log_inter[-2][-2] # multiplier for ce -> lmbda_k ; alm4sqp에서 수렴된 라그랑주 승수를 리스트에 담을 때 수렴이 이미 됐는데 한 번 더 업데이트해서 담기 때문에 마지막 담긴 값 직전의 값이 실제 수렴 라그랑주 승수임
        nu_new = np.maximum(log_inter[-1][-2], 0) # multiplier for ci -> nu_k (nonnegativity) ; alm4sqp에서 수렴된 라그랑주 승수를 리스트에 담을 때 수렴이 이미 됐는데 한 번 더 업데이트해서 담기 때문에 마지막 담긴 값 직전의 값이 실제 수렴 라그랑주 승수임
        if k == 0:
            mu = 1
        alpha, mu = steplength_merit(k, mu, f, ce, ce_torch, ci, ci_torch, x_k, p_new, grad_f_k, B_k, lmbda_k=lmbda_new, nu_k=nu_new) # step length alpha_k
        x_new = x_k + alpha*p_new # new point x_k+1
        f_new = f(x_new) # new f value f_k+1
        ce_new = [ce_j(x_new) for ce_j in ce] # ce_k+1
        ci_new = [ci_j(x_new) for ci_j in ci] # ci_k+1
        grad_f_new = grad_ad(f_torch, x_new) # ∇f_k+1
        grad_ce_new = [grad_ad(ce_torch_j, x_new) for ce_torch_j in ce_torch] # ∇ce_k+1
        grad_ci_new = [grad_ad(ci_torch_j, x_new) for ci_torch_j in ci_torch] # ∇ci_k+1

        # ---------------- Ingredients for calculating hessian_pd_k(BFGS) ----------------
        s_k = x_new - x_k # s_k (BFGS)
        # Lagrangian gradient(∇L(x_k) = ∇f(x_k) - Σλ_k∇ce(x_k) - Συ_k∇ci(x_k)) diff (BFGS)
        grad_L_new = grad_f_new \
            - sum(lmbda_new[j]*grad_ce_new[j] for j in range(len(ce))) \
            - sum(nu_new[j]*grad_ci_new[j] for j in range(len(ci)))
        grad_L_cur = grad_f_k \
            - sum(lmbda_new[j]*grad_ce_k[j] for j in range(len(ce))) \
            - sum(nu_new[j]*grad_ci_k[j] for j in range(len(ci)))
        y_k = grad_L_new - grad_L_cur # y_k (BFGS)

        # ---------------- Convergence / Termination Criteria ----------------
        # residuals
        r_grad_L  = np.linalg.norm(grad_L_new) # ‖∇L(x_k+1)‖
        r_ce   = np.max(np.abs(ce_new)) # ‖ce(x_k+1)‖∞
        r_ci   = np.max(np.maximum(-np.array(ci_new), 0.0)) # ‖ci(x_k+1)‖∞
        r_step = np.linalg.norm(x_new - x_k) # ‖∆x‖

        list_x.append(x_new)
        list_f.append(f_new)
        list_grad_f.append(grad_f_new) # 굳이 저장 안 해도 됨(생략 가능)
        list_grad_L.append(grad_L_new)
        list_ce.append(ce_new)
        list_ci.append(ci_new)
        list_lmbda.append(lmbda_new)
        list_nu.append(nu_new)

        # convergence check
        if ((r_grad_L   <= tol_opt_final and
            r_ce    <= tol_eq_final  and
            r_ci    <= tol_ineq_final and
            r_step  <= tol_step_final * (1.0 + np.linalg.norm(x_new))) | 
            ((k >= 5) & (r_step < tol_step_final))):
            print()
            break

        # 현황 확인
        x_str = ", ".join([f"{xi:.8f}" for xi in x_new])
        print("\n log - SQP")
        print(f"x{k+1:02d} = [{x_str}] \n"
            f"‖∆x‖ = {r_step:.2e}, "
            f"f = {f_new:.4e}, "
            f"‖∇L‖ = {r_grad_L:.2e}, "
            f"‖ce‖∞ = {r_ce:.2e}, "
            f"‖ci‖∞ = {r_ci:.2e}, "
            f"‖λ‖ = {np.linalg.norm(lmbda_new):.2e}, "
            f"‖ν‖ = {np.linalg.norm(nu_new):.2e} \n")

        # optional safety stop
        if not np.isfinite(x_new).all() or not np.isfinite(f_new):
            print("NaN or Inf detected — terminating early.")
            break
    else:
        print("\nReached maximum outer iterations (no convergence).")
    # ---------------- Output summary ----------------
    print(f"Final iterate: x* = {x_new}")
    print(f"f(x*)         = {f_new}")
    print(f"λ*            = {lmbda_new}")
    print(f"ν*            = {nu_new}")
    print(f"‖∇L(x*)‖     = {r_grad_L:.3e}")
    print(f"‖ce(x*)‖∞   = {r_ce:.3e}")
    print(f"‖ci(x*)‖∞ = {r_ci:.3e}")
    
    log = [list_x, list_f, list_grad_f, list_grad_L, list_ce, list_ci, list_lmbda, list_nu]
    
    return log