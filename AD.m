classdef AD
    %AD Automatic Differentiation Class in forward mode
    %
    
    properties
        x
        dx
    end
    
    methods
        
        %
        % Constructor
        %   Possibilities:
        %       - AD()
        %       - AD(10) // A constant
        %       - AD(X) // x is an AD object
        %       - AD(10,0) // A value, with the value of the derivative
        %

        function obj = AD(x,dx)
            if( nargin == 0)
                obj.x=0;
                obj.dx=0;
            elseif ( nargin==1 )
                obj.x= x;
                obj.dx= eye(length(x));
            elseif(isa(x,'AD'))
                obj=x;
            elseif( nargin == 2)
                obj.x=x;
                obj.dx=dx;
            end
        end
        
        %
        % Basic Operators:
        %       
            
        function obj = minus(a,b)
            obj = plus(a,-b);
%             if(~isa(a,'AD') && isa(b,'AD'))
%                 obj = AD( a  - b.x, b.dx );
%             elseif(isa(a,'AD') && ~isa(b,'AD'))
%                 obj = AD( a.x  - b, a.dx );
%             else
%                 obj = AD( a.x - b.x, a.dx - b.dx );
%             end
        end
        
        function obj = plus(a,b)
            if(~isa(a,'AD') && isa(b,'AD'))
                obj = AD( a  + b.x, b.dx );
            elseif(isa(a,'AD') && ~isa(b,'AD'))
                obj = AD( a.x  + b, a.dx );
            elseif(~isa(a,'AD') && ~isa(b,'AD'))
                obj = AD( a.x  + b.x, a.dx + b.dx );
            end
        end
        
        function obj = times(a,b)
            if(~isa(a,'AD') && isa(b,'AD'))
                obj = AD( a.*b.x, a.*b.dx );
            elseif(isa(a,'AD') && ~isa(b,'AD'))
                obj = AD( a.x.*b, a.dx.*b );
            elseif(~isa(a,'AD') && ~isa(b,'AD'))
                obj = AD( a.x.*b.x, a.x.*b.dx + a.dx.*b.x );
            end
        end
        
        function obj = mtimes(a,b)
            if(~isa(a,'AD') && isa(b,'AD'))
                obj = AD( a.*b.x, a.*b.dx );
            elseif(isa(a,'AD') && ~isa(b,'AD'))
                obj = AD( a.x.*b, a.dx.*b );
            elseif(~isa(a,'AD') && ~isa(b,'AD'))
                obj = AD( a.x.*b.x, a.x.*b.dx + a.dx.*b.x );
            end
        end
        
        function obj = uminus(a)
            obj = AD( -a.x, -a.dx );  
        end
    
        function obj = uplus(a)
            obj = a;
        end
        
        function obj = power(a,b)
            if(~isa(a,'AD') && isa(b,'AD'))
                obj = AD(a.^b.x , ...
                        (a.^b.x.*log(a) ).* b.dx);
            elseif(isa(a,'AD') && ~isa(b,'AD'))
                obj = AD(a.x.^b, ...
                       (b.*a.x.^(b-1)).* a.dx);
            elseif(~isa(a,'AD') && ~isa(b,'AD'))
                obj = AD( a.x.^b.x,  ...
                  (b.x.*a.x).^(b.x-1)*a.dx + a.x.^b.x .*b.dx * log(a.x) );
            end
        end

        
        function obj = mpower(a,b)
            if(~isa(a,'AD') && isa(b,'AD'))
                obj = AD(a.^b.x , ...
                        (a.^b.x.*log(a) ).* b.dx);
            elseif(isa(a,'AD') && ~isa(b,'AD'))
                obj = AD(a.x.^b, ...
                       (b.*a.x.^(b-1)).* a.dx);
            elseif(~isa(a,'AD') && ~isa(b,'AD'))
                obj = AD( a.x.^b.x,  ...
                  (b.x.*a.x).^(b.x-1)*a.dx + a.x.^b.x .*b.dx * log(a.x) );
            end
        end        
        
        
        function obj = transpose(a)
            obj = AD( transpose(a.x), transpose(a.dx) );  
        end
        
        %
        % Trigonometric Functions
        %
        
        function obj = acos(a)
            obj = AD( acos(a.x), -a.dx./sqrt(1-a.x.^2) );
        end
        
        function obj = acosh(a)
            obj = AD( acosh(a.x), a.dx./sqrt(a.x.^2-1) );
        end
        
        function obj = acot(a)
            obj = AD( acot(a.x), -a.dx./(a.x.^2+1) );
        end
        
        function obj = acoth(a)
            obj = AD( acoth(a.x), -a.dx.*csch(a.x).^2 );
        end
        
        function obj = acsc(a)
            obj = AD( acsc(a.x), -a.dx./(a.x.*sqrt(a.x.^2-1)) );
        end
        
        function obj = sec(a)
          v   = sec(a.x);
          obj = AD( v, v.*tan(a.x).*a.dx );
        end

        function obj = sech(a)
          v   = sech(a.x);
          obj = AD( v, -tanh(a.x).*v.*a.dx );
        end

        function obj = sin(a)
          obj = AD( sin(a.x), cos(a.x).*a.dx );
        end

        function obj = sinh(a)
          obj = AD( sinh(a.x), cosh(a.x).*a.dx );
        end

        function obj = tan(a)
          obj = AD( tan(a.x), (sec(a.x).^2).*a.dx );
        end

        function obj = tanh(a)
          obj = AD( tanh(a.x), (sech(a.x).^2).*a.dx );
        end
        
        function obj = log(a)
           obj = AD(log(a.x), 1./(a.x.*a.dx));
        end
        
        %
        % Misc. Functions
        %
        
        function sub  = subsref(a,s)
            sub = AD(a.x(s.subs{1}), a.dx(s.subs{1},:));
        end
        
        function [] = disp(c)
            fprintf('x : \n');
            fprintf([repmat('%f\t', 1, size(c.x, 2)) '\n'], c.x)
            fprintf('\n')
            fprintf('dx : \n');
            fprintf([repmat('%f\t', 1, size(c.dx, 2)) '\n'], c.dx)
            fprintf('\n')
        end
    end
    
end

