local M = {}

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 ResNet fine-tuning script')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-data',       './dataset/',         'Path to dataset')
   cmd:option('-manualSeed', 42,          'Manually set RNG seed')
   cmd:option('-precision', 'single',    'Options: single | double | half')
   ------------- Data options ------------------------
   cmd:option('-nThreads',        2, 'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',         200,       'Number of total epochs to run')
   cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
   cmd:option('-testOnly',        'false', 'Run on validation set only')
   cmd:option('-validationSize',   0.3,    'part of training set to be used as validation')
   ------------- Checkpointing options ---------------
   cmd:option('-save',            './checkpoints/', 'Directory in which to save checkpoints')
   cmd:option('-resume',          'none',        'Resume from the latest checkpoint in this directory')
   cmd:option('-log',             'false',        'log training and validation data')
   ---------- Optimization options ----------------------
   cmd:option('-LR',              1e-4,   'initial learning rate')
   cmd:option('-nClasses',         2,      'Number of classes in the dataset')
   cmd:text()

   local opt = cmd:parse(arg or {})

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end

   if opt.precision == nil or opt.precision == 'single' then
      opt.tensorType = 'torch.FloatTensor'
   elseif opt.precision == 'double' then
      opt.tensorType = 'torch.DoubleTensor'
   else
      cmd:error('unknown precision: ' .. opt.precision)
   end

   return opt
end

return M
