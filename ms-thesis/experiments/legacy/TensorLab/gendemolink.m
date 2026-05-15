function url = gendemolink(name, text)
%GENDEMOLINK Generate a weblink for a demo
    
% Author(s): Nico Vervliet       (Nico.Vervliet@esat.kuleuven.be)
%
% Version History:
% - 2015/07/10   NV      Initial version
    
    baseurl = 'http://www.tensorlab.net';
    url = sprintf('<a href="matlab: web(''%s/demos/%s.html'', ''-browser'')">%s</a>', ...
                  baseurl, name, text);
end
