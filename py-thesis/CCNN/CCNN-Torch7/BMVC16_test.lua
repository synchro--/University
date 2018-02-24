require 'torch'   -- torch
require 'nn'      -- provides a normalization operator
require 'optim'
require 'image'

if not opt then
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model','CCNN_BMVC16.net','network name')
   cmd:option('-disparity','disparity.png','input disparity map')
   cmd:option('-output','confidence','output name')
   cmd:option('-enable16Bit','true','Enable 16 bit precision')
   cmd:text()
   opt = cmd:parse(arg or {})
end

torch.setdefaulttensortype('torch.FloatTensor') -- preprocessing requires a floating point representation 

function fully_connected_to_fully_convolutional(model)
	local fcnet = nn.Sequential()
	
	for i, v in ipairs(model.modules) do 
		if torch.type(v) == 'nn.Linear' then
			-- create fully convolutional layer
			local ninput = v.weight:size()[2]
			local noutput = v.weight:size()[1]
			local fclayer = nn.SpatialConvolution(ninput, noutput, 1, 1, 1, 1)
			fclayer.weight = torch.reshape(v.weight, ninput*noutput, 1, 1) 
			fclayer.bias = v.bias 
			fcnet:add(fclayer)
		else
			if torch.type(v) ~= 'nn.Reshape' then
				fcnet:add(v)	
			end
		end
	end
	return fcnet
end

function singleForwardConfidenceMap(dispImage, mapsName, enable16Bit)

	print(sys.COLORS.red..'Loading disparity tensor...')

	disparity = image.load(dispImage)

	padder_left = nn.Padding(2, -4, 8, 0) 
	padder_right = nn.Padding(2, 4, 8, 0) 
	padder_up = nn.Padding(1, -4, 8, 0) 
	padder_down = nn.Padding(1, 4, 8, 0) 

	input = padder_left:forward(disparity)
	input = padder_right:forward(input)
	input = padder_up:forward(input)
	input = padder_down:forward(input)


	model:float()
	model:evaluate()
	
	output = model:forward(input:float())

	print(sys.COLORS.yellow..'Confidence Tensor ready!')

	-- to save as 16 bit image (higher precision)
	if enable16Bit == 'true' then
		mapH = torch.FloatTensor(output:size()[2], output:size()[3])
		mapL = torch.FloatTensor(output:size()[2], output:size()[3])

		confidence = output * (256*256-1)
		mapH = (confidence / 256) / 256
		mapL = torch.mod(confidence, 256) / 256

		-- saving 16 bit as higher and lower bytes
		image.save(mapsName..'_H.png', mapH)
		image.save(mapsName..'_L.png', mapL)

		-- merging higher and lower bytes
		os.execute('./merge '..mapsName..'_H.png '..mapsName..'_L.png '..mapsName..'.png')
		os.execute('rm '..mapsName..'*_H.png')
		os.execute('rm '..mapsName..'*_L.png')
	else 

		-- saving as 8 bit image (losing precision)
		image.save(mapsName..'.png', output)
	end	
	

print(sys.COLORS.green..'Confidence Tensor saved!')
end

-- main 
net = torch.load(opt.model, 'ascii')
model = fully_connected_to_fully_convolutional(net)
singleForwardConfidenceMap(opt.disparity, opt.output, opt.enable16Bit)




