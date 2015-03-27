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
        %       - AD(10)
        %       - AD(X) //x is an AD object
        %       - AD(10,0)
        %

        function obj = AD(c,dx)
            if( nargin == 0)
                obj.x=0;
                obj.dx=0;
            elseif ( nargin==1 )
                obj.x= c;
                obj.dx= 0*c;
            elseif(isa(c,'AD'))
                obj=c;
            elseif( nargin == 2)
                obj.x=c;
                obj.dx=dx;
            end
        end
        
        %
        % Basic Operators:
        %
        %       
            
        function obj = minus(a,b)
            obj = AD( a.x - b.x, a.dx - b.dx );
        end
        
        function obj = plus(a,b)
            obj = AD( a.x  + b.x, a.dx + b.dx );
        end
        
        function obj = times(a,b)
            obj = AD( a.x.*b.x, a.x.*b.dx + a.dx.*b.x );
        end
        
        function obj = uminus(a)
            obj = AD( -a.x, -a.dx );  
        end
    
        function obj = uplus(a)
            obj = a;
        end
        
        function obj = transpose(a)
            obj = AD( transpose(a.x), transpose(a.dx) );  
        end
        
        %
        % Trigonometric Functions
        %
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
            obj = AD( acsc(a.x), -a.dx./( a.x.*sqrt(a.x.^2-1) ) );
        end
        
        function obj = sec(a)
          v   = sec(a.x);
          obj = AD( v, v.*tan(a.x).*a.dx );
        end

        function obj = sech(c)
          v   = sech(c.x);
          obj = Deriv( v, -tanh(c.x).*v.*c.dx );
        end

        function obj = sin(c)
          obj = Deriv( sin(c.x), cos(c.x).*c.dx );
        end

        function obj = sinh(c)
          obj = Deriv( sinh(c.x), cosh(c.x).*c.dx );
        end

        function obj = tan(c)
          obj = Deriv( tan(c.x), (sec(c.x).^2).*c.dx );
        end

        function obj = tanh(c)
          obj = Deriv( tanh(c.x), (sech(c.x).^2).*c.dx );
        end
        
        
        %
        % Misc. Functions
        %
        %

        function [] = disp(c)
            fprintf('\n')
            fprintf(' x : %f \n',c.x);
            fprintf('dx : %f \n',c.dx);
            fprintf('\n')
        end
    end
    
end

