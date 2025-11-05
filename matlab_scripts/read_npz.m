function [fields, T, symmetry, params] = read_npz(filename)
  np = py.importlib.import_module('numpy');
  
  data = np.load(filename);

   
  fields = py2mat(data.get('fields'));
  T      = py2mat(data.get('T'));

  symmetry.sx = py2mat(data.get('sx'));

  %Other useful parameters
  params.b0    = py2mat(data.get('b0'));
  params.nu    = py2mat(data.get('nu'));
  params.eta   = py2mat(data.get('eta'));
  params.steps = py2mat(data.get('steps'));
end

function A = py2mat(f)
    %initial version of this is ChatGPT garbage.
    import matlab.lang.makeValidName

    % Handle scalar (0-D array)
    if isempty(cell(f.shape))
        A = double(f.item());
        return
    end

    % Flatten array efficiently using nditer
    A = double(py.array.array('d', py.numpy.nditer(f)));

    % Restore original shape
    shape = fliplr(cellfun(@int64, cell(f.shape)));
    if numel(shape) > 1
      A = reshape(A, shape);
    end
    % Here lies a very dumb idea from aritficial stupidity.
    % Why would I permute the indices of my tensor? I have them ordered in
    % memory for good reason.
    %
    % Permute to fix row-major (NumPy) -> column-major (MATLAB)
    %ndims(A)
    %A = permute(A, ndims(f):-1:1);
end