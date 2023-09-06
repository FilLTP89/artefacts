currentPath = pwd;

disp(['The current path is: ', currentPath]);



images = 0:871;
datatype = 'uint16';
im_size = [400,400];
endianness = 'b';
filename = "../data/No_metal/acquisition_0/IE1705794_P406.i18%04d.raw";

sino = zeros(400,400,871);

for i = images
    im_name = sprintf(filename,i);

    fid = fopen(im_name, 'r');

    % read the data
    V = fread(fid, inf, ['*' datatype], 0, endianness);
    fclose(fid);

    % shape it
    sino(:,:,i+1) = reshape(V, im_size);
end

% Get background intensity
intensity = mean(sino(end-20:end,100:300,350),'all');

sino = -log(sino/intensity);
sino(sino < 0) = 0;

save("sino","sino","-v7.3")
